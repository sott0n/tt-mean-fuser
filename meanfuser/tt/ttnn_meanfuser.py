# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MeanFuser on TTNN: one device-resident graph from the stitched camera image to the
normalised ARM trajectory. Only the input upload and the final cumsum/atan2 run on host.

Reuses the DiffusionDrive ResNet-34 blocks and GPT fusion. MeanFuser-specific choices:
  * The LiDAR branch input is a learned constant, so its stem and layer1 are computed once
    at build time.
  * bev_query needs the 64×64 FPN map and the bilinearly upsampled 8×8 key-val grid only at
    even pixels (nearest 0.5× downsample). up_conv4 therefore runs with stride 2, and the
    kv upsample + subsample is a fixed (1024, 64) interpolation matrix.
  * The MeanFlow decoder has one query token per proposal and a memory shared by all
    proposals, so the K proposals run as one length-K query sequence. Its self-attention
    over a single token reduces to out_proj(v_proj(x)).
  * head_dim is 16, below the SDPA minimum, so attention is explicit matmul + softmax.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn
from models.experimental.diffusion_drive.tt.common import fold_bn
from models.experimental.diffusion_drive.tt.ttnn_gpt_fusion import TtnnFuseFeatures
from models.experimental.diffusion_drive.tt.ttnn_resnet34 import prep_conv_weights
from models.experimental.meanfuser.reference.model import (
    ACTION_DIM_DELTA,
    HORIZON,
    MeanFuserModel,
    cumsum_traj,
)


# The camera is uploaded L1 height-sharded with RGB padded to 8 channels (16 B rows); from a
# DRAM-interleaved 3-channel tensor the stem conv spends ~0.45 ms resharding its input.
CAMERA_CHANNELS = 8
CAMERA_CORES = 64


def _tile(t: torch.Tensor, device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    return ttnn.from_torch(t.detach().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _linear_params(linear: nn.Linear, device) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]:
    w = _tile(linear.weight.T.float(), device)
    b = _tile(linear.bias.reshape(1, -1).float(), device) if linear.bias is not None else None
    return w, b


def _ln_params(ln: nn.LayerNorm, device):
    return _tile(ln.weight.reshape(1, -1), device), _tile(ln.bias.reshape(1, -1), device), float(ln.eps)


def _layer_norm(x, p):
    return ttnn.layer_norm(x, weight=p[0], bias=p[1], epsilon=p[2])


def _interleaved_tile(x: ttnn.Tensor) -> ttnn.Tensor:
    if x.is_sharded():
        x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


def _bilinear_matrix(n_in: int, n_out: int) -> torch.Tensor:
    """(n_out, n_in) weights of 1-D bilinear resize with align_corners=False."""
    eye = torch.eye(n_in).reshape(n_in, 1, n_in)
    return F.interpolate(eye, size=n_out, mode="linear", align_corners=False).reshape(n_in, n_out).T


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class _Attention:
    """nn.MultiheadAttention (batch_first, no mask) on SDPA.

    head_dim is 16, below the SDPA minimum, so every head is zero-padded to 32 in the Q/K/V
    projections and the matching out_proj rows. The padded lanes add 0 to q·k and produce 0
    outputs that out_proj ignores; the softmax scale stays 1/sqrt(16).
    """

    PAD_HEAD_DIM = 32

    def __init__(self, mha: nn.MultiheadAttention, device) -> None:
        E = mha.embed_dim
        nh, hd, hp = mha.num_heads, E // mha.num_heads, self.PAD_HEAD_DIM
        self._nh = nh
        W, b = mha.in_proj_weight.detach(), mha.in_proj_bias.detach()

        def pad_heads(w, bias):
            # (E_out, E_in) rows grouped by head -> (nh*hp, E_in) with zero rows per head.
            wp = F.pad(w.reshape(nh, hd, -1), (0, 0, 0, hp - hd)).reshape(nh * hp, -1)
            bp = F.pad(bias.reshape(nh, hd), (0, hp - hd)).reshape(1, -1)
            return wp, bp

        wq, bq = pad_heads(W[:E], b[:E])
        wk, bk = pad_heads(W[E : 2 * E], b[E : 2 * E])
        wv, bv = pad_heads(W[2 * E :], b[2 * E :])
        self._wq, self._bq = _tile(wq.T, device), _tile(bq, device)
        # Fused projections feed split_query_key_value_and_split_heads, which needs three
        # same-length sections: [Q|K|V] for self-attention, [K|K|V] (first copy unused) for memory.
        self._wqkv = _tile(torch.cat([wq, wk, wv], 0).T, device)
        self._bqkv = _tile(torch.cat([bq, bk, bv], 1), device)
        self._wkkv = _tile(torch.cat([wk, wk, wv], 0).T, device)
        self._bkkv = _tile(torch.cat([bk, bk, bv], 1), device)
        wo = F.pad(mha.out_proj.weight.detach().reshape(E, nh, hd), (0, hp - hd)).reshape(E, nh * hp)
        self._wo, self._bo = _tile(wo.T, device), _tile(mha.out_proj.bias.reshape(1, -1), device)
        self._scale = 1.0 / math.sqrt(hd)

    def _heads(self, x: ttnn.Tensor, B: int, S: int) -> ttnn.Tensor:
        return ttnn.permute(ttnn.reshape(x, (B, S, self._nh, self.PAD_HEAD_DIM)), (0, 2, 1, 3))

    def _split(self, fused: ttnn.Tensor):
        return ttnn.transformer.split_query_key_value_and_split_heads(fused, num_heads=self._nh, transpose_key=False)

    def project_kv(self, mem: ttnn.Tensor, B: int, S: int):
        """mem (B, S, E) -> padded k, v heads (B, nh, S, 32)."""
        _, k, v = self._split(ttnn.linear(mem, self._wkkv, bias=self._bkkv))
        return k, v

    def _attend(self, q, k, v) -> ttnn.Tensor:
        o = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, scale=self._scale)
        return ttnn.linear(ttnn.transformer.concatenate_heads(o), self._wo, bias=self._bo)

    def self_attention(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self._attend(*self._split(ttnn.linear(x, self._wqkv, bias=self._bqkv)))

    def __call__(self, x: ttnn.Tensor, B: int, Sq: int, kv) -> ttnn.Tensor:
        q = self._heads(ttnn.linear(x, self._wq, bias=self._bq), B, Sq)
        return self._attend(q, *kv)


class _DecoderLayer:
    """nn.TransformerDecoderLayer (post-norm, ReLU, dropout 0)."""

    def __init__(self, layer: nn.TransformerDecoderLayer, device, single_token_self_attn: bool = False) -> None:
        self._single = single_token_self_attn
        if single_token_self_attn:
            # softmax over one key is 1, so self-attention is out_proj(v_proj(x)); fold both in fp32.
            sa = layer.self_attn
            E = sa.embed_dim
            Wv, bv = sa.in_proj_weight.detach()[2 * E :], sa.in_proj_bias.detach()[2 * E :]
            Wo, bo = sa.out_proj.weight.detach(), sa.out_proj.bias.detach()
            self._sa_w = _tile((Wo @ Wv).T, device)
            self._sa_b = _tile((Wo @ bv + bo).reshape(1, -1), device)
        else:
            self._sa = _Attention(layer.self_attn, device)
        self._ca = _Attention(layer.multihead_attn, device)
        self._l1 = _linear_params(layer.linear1, device)
        self._l2 = _linear_params(layer.linear2, device)
        self._n1 = _ln_params(layer.norm1, device)
        self._n2 = _ln_params(layer.norm2, device)
        self._n3 = _ln_params(layer.norm3, device)
        self._const_x = None  # norm1(x + self_attn(x)) when x is input-independent

    def __call__(self, x, B: int, Sq: int, mem_kv) -> ttnn.Tensor:
        if self._const_x is not None:
            x = self._const_x
        else:
            if self._single:
                sa = ttnn.linear(x, self._sa_w, bias=self._sa_b)
            else:
                sa = self._sa.self_attention(x)
            x = _layer_norm(ttnn.add(x, sa), self._n1)
        x = _layer_norm(ttnn.add(x, self._ca(x, B, Sq, mem_kv)), self._n2)
        ff = ttnn.linear(x, self._l1[0], bias=self._l1[1], activation="relu")
        ff = ttnn.linear(ff, self._l2[0], bias=self._l2[1])
        return _layer_norm(ttnn.add(x, ff), self._n3)


class _Decoder:
    def __init__(
        self,
        decoder: nn.TransformerDecoder,
        device,
        single_token_self_attn: bool = False,
        const_query: Optional[torch.Tensor] = None,
    ) -> None:
        """const_query: the decoder input when it is a constant; layer 0's self-attention block
        then depends on nothing at runtime and is computed once in fp32."""
        self._layers = [_DecoderLayer(l, device, single_token_self_attn) for l in decoder.layers]
        if const_query is not None:
            l0 = decoder.layers[0]
            with torch.no_grad():
                sa = l0.self_attn(const_query, const_query, const_query, need_weights=False)[0]
                self._layers[0]._const_x = _tile(l0.norm1(const_query + sa), device)

    def __call__(self, x, memory, B: int, Sq: int, Skv: int) -> ttnn.Tensor:
        for layer in self._layers:
            x = layer(x, B, Sq, layer._ca.project_kv(memory, B, Skv))
        return x


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class _Conv:
    """Plain / BN-folded Conv2d with device-cached weights. NHWC-flat (1,1,B*H*W,C) in and out.

    HiFi2 with fp32 accumulation: with the default conv compute config the error compounds through
    the ResNet-34 stages (deepest LiDAR feature PCC ~0.97, trajectory off by ~0.7 m). HiFi2 matches
    HiFi4 on navtest.
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        bn: Optional[nn.BatchNorm2d] = None,
        stride: Optional[int] = None,
        pad_in_channels: Optional[int] = None,
    ) -> None:
        if bn is not None:
            w, b = fold_bn(conv, bn)
        else:
            w = conv.weight.detach()
            b = conv.bias.detach() if conv.bias is not None else torch.zeros(conv.out_channels)
        self.cin, self.cout = conv.in_channels, conv.out_channels
        if pad_in_channels is not None:
            w = F.pad(w, (0, 0, 0, 0, 0, pad_in_channels - self.cin))
            self.cin = pad_in_channels
        self._w, self._b = prep_conv_weights(w.to(torch.bfloat16), b.to(torch.bfloat16))
        self._k = int(conv.kernel_size[0])
        self._s = int(stride if stride is not None else conv.stride[0])
        self._p = int(conv.padding[0])

    def __call__(self, device, x, B: int, H: int, W: int, relu: bool = True, sharded_out: bool = False):
        if sharded_out:
            # Keep a ResNet stage in L1. Otherwise conv2d runs DRAM-sliced and wraps every conv
            # in an interleaved<->sharded conversion.
            # Height sharding spreads small outputs over too few cores (e.g. 8 for 8×32 px).
            out_px = B * ((H + 2 * self._p - self._k) // self._s + 1) * ((W + 2 * self._p - self._k) // self._s + 1)
            layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED if out_px >= 2048 else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        else:
            layout = None
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            deallocate_activation=False,
            reallocate_halo_output=True,
            reshard_if_not_optimal=sharded_out,
            shard_layout=layout,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if relu else None,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            math_approx_mode=False,
        )
        out, [Ho, Wo], [self._w, self._b] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self._w,
            bias_tensor=self._b,
            in_channels=self.cin,
            out_channels=self.cout,
            device=device,
            kernel_size=[self._k, self._k],
            stride=[self._s, self._s],
            padding=[self._p] * 4,
            dilation=[1, 1],
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=conv_config,
            compute_config=compute_config,
            slice_config=ttnn.Conv2dL1FullSliceConfig if sharded_out else None,
            return_weights_and_bias=True,
            return_output_dim=True,
        )
        return (out if sharded_out else _interleaved_tile(out)), Ho, Wo


class _BasicBlock:
    """timm ResNet BasicBlock: relu(bn2(conv2(relu(bn1(conv1 x)))) + shortcut(x))."""

    def __init__(self, block, device) -> None:
        self._d = device
        self._c1 = _Conv(block.conv1, block.bn1)
        self._c2 = _Conv(block.conv2, block.bn2)
        self._ds = _Conv(block.downsample[0], block.downsample[1]) if block.downsample is not None else None

    def __call__(self, x, shape):
        """Output stays in the conv's sharded L1 layout; the stage caller interleaves once at the end."""
        B, H, W, _ = shape
        y, H1, W1 = self._c1(self._d, x, B, H, W, sharded_out=True)
        y, H2, W2 = self._c2(self._d, y, B, H1, W1, relu=False, sharded_out=True)
        sc = self._ds(self._d, x, B, H, W, relu=False, sharded_out=True)[0] if self._ds is not None else x
        if sc.memory_config() != y.memory_config():
            sc = ttnn.to_memory_config(sc, y.memory_config())
        y = ttnn.add_(y, sc, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)])
        return y, (B, H2, W2, self._c2.cout)


def _run_stage(blocks: List[_BasicBlock], x, shape):
    for blk in blocks:
        x, shape = blk(x, shape)
    return _interleaved_tile(x), shape


class _SdpaSelfAttn:
    """GPT SelfAttention as fused QKV + head split + SDPA. Needs head_dim >= 32."""

    def __init__(self, sa, device) -> None:
        self._nh = sa.n_head
        w = torch.cat([sa.query.weight, sa.key.weight, sa.value.weight], 0).T
        b = torch.cat([sa.query.bias, sa.key.bias, sa.value.bias]).reshape(1, -1)
        self._wqkv, self._bqkv = _tile(w, device), _tile(b, device)
        self._wo, self._bo = _linear_params(sa.proj, device)

    def __call__(self, x, B, T, C):
        qkv = ttnn.linear(x, self._wqkv, bias=self._bqkv)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=self._nh, transpose_key=False)
        o = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)
        return ttnn.linear(ttnn.transformer.concatenate_heads(o), self._wo, bias=self._bo)


class _FuseFeatures(TtnnFuseFeatures):
    """DiffusionDrive GPT fusion kept in TILE layout end to end (batch 1).

    * avg_pool2d runs on the TILE input and the bilinear upsample is a matmul with a fixed matrix.
      The ROW_MAJOR pool/upsample path needs 6-7 layout ops per resample and is 1.5-5x slower.
    * Scales with head_dim >= 32 use SDPA; the per-head reshape/permute path is 3-5x slower.
    """

    def __init__(self, ref_backbone, device) -> None:
        super().__init__(ref_backbone, device)
        for gpt_ref, gpt in zip(ref_backbone.transformers, self._gpt):
            for blk_ref, blk in zip(gpt_ref.blocks, gpt._blocks):
                if blk_ref.attn.key.out_features // blk_ref.attn.n_head >= 32:
                    blk._attn = _SdpaSelfAttn(blk_ref.attn, device)

        cfg = ref_backbone.config
        self._mm_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        # Pooled + projected LiDAR tokens for scales whose LiDAR input is input-independent.
        self.const_lid_tokens = {}
        self._tile_resample = {}
        for i in range(len(self._gpt)):
            hi, wi = cfg.camera_height // (4 << i), cfg.camera_width // (4 << i)
            hl, wl = cfg.lidar_resolution_height // (4 << i), cfg.lidar_resolution_width // (4 << i)
            self._tile_resample[i] = (
                self._resample_mats(hi, wi, self._iv, self._ih, device),
                self._resample_mats(hl, wl, self._lv, self._lh, device),
            )

    @staticmethod
    def _resample_mats(H, W, v, h, device):
        """(H, W, v, h, bilinear v×h -> H×W upsample matrix over row-major-flattened pixels).
        None when the map is already at anchor size."""
        if (H, W) == (v, h):
            return None
        up = torch.kron(_bilinear_matrix(v, H), _bilinear_matrix(h, W))  # (H*W, v*h)
        return H, W, v, h, _tile(up[None], device)

    def _pool_tokens(self, x, mats):
        """x (1, H*W, C) TILE -> (1, v*h, C) TILE. avg_pool2d takes TILE input directly."""
        if mats is None:
            return x
        H, W, v, h, _ = mats
        C = x.shape[-1]
        out = ttnn.avg_pool2d(
            ttnn.reshape(x, (1, 1, H * W, C)),
            batch_size=1,
            input_h=H,
            input_w=W,
            channels=C,
            kernel_size=[H // v, W // h],
            stride=[H // v, W // h],
            padding=[0, 0],
        )
        return ttnn.reshape(_interleaved_tile(out), (1, v * h, C))

    def _upsample_add(self, x, tokens, mats):
        up = tokens if mats is None else ttnn.matmul(mats[4], tokens, compute_kernel_config=self._mm_cfg)
        return ttnn.add(x, up)

    def forward_dev(self, img_t, img_shape, lid_t, lid_shape, layer_idx: int):
        img_m, lid_m = self._tile_resample[layer_idx]
        B, Hi, Wi, Ci = img_shape
        _, Hl, Wl, Cl = lid_shape
        assert B == 1
        img = ttnn.reshape(img_t, (1, Hi * Wi, Ci))
        lid = ttnn.reshape(lid_t, (1, Hl * Wl, Cl))
        n_lid = self._lv * self._lh
        l2i_w, l2i_b = self._l2i[layer_idx]
        if layer_idx in self.const_lid_tokens:
            lid_tok = self.const_lid_tokens[layer_idx]
        else:
            lid_tok = ttnn.linear(self._pool_tokens(lid, lid_m), l2i_w, bias=l2i_b)
        tokens = ttnn.concat([self._pool_tokens(img, img_m), lid_tok], dim=1)
        x = self._gpt[layer_idx](tokens, 1, self._n_img + n_lid, Ci)
        img_x = ttnn.slice(x, [0, 0, 0], [1, self._n_img, Ci])
        lid_x = ttnn.slice(x, [0, self._n_img, 0], [1, self._n_img + n_lid, Ci])
        i2l_w, i2l_b = self._i2l[layer_idx]
        lid_x = ttnn.linear(lid_x, i2l_w, bias=i2l_b)
        img = self._upsample_add(img, img_x, img_m)
        lid = self._upsample_add(lid, lid_x, lid_m)
        return (
            ttnn.reshape(img, (1, 1, Hi * Wi, Ci)),
            img_shape,
            ttnn.reshape(lid, (1, 1, Hl * Wl, Cl)),
            lid_shape,
        )


class TtnnMeanFuserBackbone:
    """TransFuser backbone with constant LiDAR latent. camera (1,1,B*H*W,3) -> (lidar 8×8×512, FPN 32×32×64)."""

    def __init__(self, ref, device, batch_size: int) -> None:
        self._d = device
        self._B = batch_size
        img = ref.image_encoder
        self._img_stem = _Conv(img.conv1, img.bn1, pad_in_channels=CAMERA_CHANNELS)
        self._img_stages: List[List[_BasicBlock]] = []
        self._lid_stages: List[List[_BasicBlock]] = []
        for i in range(4):
            self._img_stages.append([_BasicBlock(b, device) for b in getattr(img, f"layer{i + 1}")])
            self._lid_stages.append([_BasicBlock(b, device) for b in getattr(ref.lidar_encoder, f"layer{i + 1}")])
        self._fusion = _FuseFeatures(ref, device)

        self._c5 = _Conv(ref.c5_conv)
        self._up5 = _Conv(ref.up_conv5)
        # Only the even pixels of up_conv4's 64×64 output are consumed downstream.
        self._up4 = _Conv(ref.up_conv4, stride=2)
        self._up2 = int(ref.upsample.scale_factor)
        self._up4_size = tuple(ref.upsample2.size)

        self._lid0, self._lid0_shape = self._lidar_layer1(ref)
        with torch.no_grad():
            lid = ref.lidar_encoder
            x = lid.layer1(lid.maxpool(lid.act1(lid.bn1(lid.conv1(ref.lidar_latent)))))
            tok = ref.lidar_channel_to_img[0](ref.avgpool_lidar(x)).flatten(2).transpose(1, 2)
        self._fusion.const_lid_tokens[0] = _tile(tok.expand(batch_size, -1, -1), device)

    def _lidar_layer1(self, ref):
        """Run the input-independent LiDAR stem + layer1 once; the result is reused every frame."""
        lid = ref.lidar_encoder
        with torch.no_grad():
            x = lid.maxpool(lid.act1(lid.bn1(lid.conv1(ref.lidar_latent))))
        x = x.expand(self._B, -1, -1, -1)
        B, C, H, W = x.shape
        t = _tile(x.permute(0, 2, 3, 1).reshape(1, 1, B * H * W, C), self._d)
        return _run_stage(self._lid_stages[0], t, (B, H, W, C))

    def _stem(self, x, B: int, H: int, W: int):
        out, Ho, Wo = self._img_stem(self._d, x, B, H, W, sharded_out=True)
        out = ttnn.max_pool2d(
            out,
            batch_size=B,
            input_h=Ho,
            input_w=Wo,
            channels=self._img_stem.cout,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
        )
        Hp, Wp = (Ho + 2 - 3) // 2 + 1, (Wo + 2 - 3) // 2 + 1
        return _interleaved_tile(out), (B, Hp, Wp, self._img_stem.cout)

    def _upsample(self, x, B: int, H: int, W: int, C: int, sh: int, sw: int):
        x = ttnn.to_layout(ttnn.reshape(x, (B, H, W, C)), ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.upsample(x, [sh, sw], mode="bilinear")
        return ttnn.reshape(_interleaved_tile(x), (1, 1, B * H * sh * W * sw, C))

    def __call__(self, camera, B: int, H: int, W: int):
        img, img_shape = self._stem(camera, B, H, W)
        lid, lid_shape = self._lid0, self._lid0_shape
        for i in range(4):
            img, img_shape = _run_stage(self._img_stages[i], img, img_shape)
            if i > 0:
                lid, lid_shape = _run_stage(self._lid_stages[i], lid, lid_shape)
            img, img_shape, lid, lid_shape = self._fusion.forward_dev(img, img_shape, lid, lid_shape, i)
        ttnn.deallocate(img)

        _, h, w, _ = lid_shape
        p5, h5, w5 = self._c5(self._d, lid, B, h, w)
        p5 = self._upsample(p5, B, h5, w5, self._c5.cout, self._up2, self._up2)
        h5, w5 = h5 * self._up2, w5 * self._up2
        p4, h4, w4 = self._up5(self._d, p5, B, h5, w5)
        s_h, s_w = self._up4_size[0] // h4, self._up4_size[1] // w4
        p4 = self._upsample(p4, B, h4, w4, self._up5.cout, s_h, s_w)
        p3, h3, w3 = self._up4(self._d, p4, B, h4 * s_h, w4 * s_w)
        return lid, lid_shape, p3, (B, h3, w3, self._up4.cout)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class TtnnMeanFuser:
    """
    Usage:
        tt = TtnnMeanFuser(reference_model, device, batch_size=1)
        out = tt(camera, status, noise)   # torch in / torch out
    """

    def __init__(self, ref: MeanFuserModel, device, batch_size: int = 1) -> None:
        self._d = device
        self._B = B = batch_size
        cfg = ref.config
        self._K = cfg.num_proposals
        d = cfg.tf_d_model
        self._dmodel = d
        self._backbone = TtnnMeanFuserBackbone(ref._backbone, device, B)

        # perception
        bd = ref._bev_downscale
        self._bd_w = _tile(bd.weight.reshape(bd.out_channels, bd.in_channels).T, device)
        self._bd_b = _tile(bd.bias.reshape(1, -1), device)
        self._status = _linear_params(ref._status_encoding, device)
        self._kv_emb = _tile(ref._keyval_embedding.weight[None], device)
        self._query = _tile(ref._query_embedding.weight[None].expand(B, -1, -1), device)
        self._n_query = ref._query_embedding.num_embeddings
        self._tf_decoder = _Decoder(
            ref._tf_decoder, device, const_query=ref._query_embedding.weight[None].expand(B, -1, -1)
        )

        # bev_query = LN(ReLU([interp(kv) | p3] @ W + b)); interp is linear, so W_kv is applied first.
        bp = ref.bev_proj[0]
        self._bp_wkv = _tile(bp.weight[:, :d].T, device)
        self._bp_wp3 = _tile(bp.weight[:, d:].T, device)
        self._bp_b = _tile(bp.bias.reshape(1, -1), device)
        self._bp_ln = _ln_params(ref.bev_proj[2], device)
        bb_cfg = ref._backbone.config
        kh, kw = bb_cfg.lidar_vert_anchors, bb_cfg.lidar_horz_anchors
        H4 = bb_cfg.lidar_resolution_height // bb_cfg.bev_down_sample_factor
        W4 = bb_cfg.lidar_resolution_width // bb_cfg.bev_down_sample_factor
        my = _bilinear_matrix(kh, H4)[::2]
        mx = _bilinear_matrix(kw, W4)[::2]
        self._n_bev = my.shape[0] * mx.shape[0]
        self._interp = _tile(torch.kron(my, mx)[None], device)  # (1, 1024, 64)
        self._n_kv_grid = kh * kw

        # MeanFlow head, one step from (r, t) = (0, 1)
        mf = ref._meanflow_head.model
        t_emb = mf.time_embedding(torch.zeros(1), torch.ones(1)).detach()  # (1, 1, d)
        pos = mf.cond_pos_emb.detach()
        n_ctx = cfg.num_bounding_boxes
        self._cond_time = _tile((t_emb + pos[:, :1]).expand(B, -1, -1), device)
        self._cond_pos_bev = _tile(pos[:, 1 : 1 + self._n_bev], device)
        self._cond_pos_ctx = _tile(pos[:, 1 + self._n_bev : 1 + self._n_bev + n_ctx], device)
        self._n_cond = 1 + self._n_bev + n_ctx
        self._mf_in = _linear_params(mf.input_emb, device)
        self._mf_pos = _tile(mf.pos_emb, device)
        self._mf_decoder = _Decoder(mf.blocks[0], device, single_token_self_attn=True)
        self._mf_lnf = _ln_params(mf.ln_f, device)
        self._mf_out = _linear_params(mf.output_emb, device)

        # ARM
        arm = ref._arm_model_head
        self._arm_enc1 = _linear_params(arm.traj_encoder[0], device)[0]
        self._arm_enc2 = _linear_params(arm.traj_encoder[2], device)[0]
        self._arm_decoder = _Decoder(arm.bev_cross_attn, device)
        self._arm_ln = _ln_params(arm.norm1, device)
        self._arm_recon = [_linear_params(arm.trajectory_recon[i], device)[0] for i in (0, 2, 4)]

    # -- device graph ------------------------------------------------------
    def _encoder(self, camera, status, H: int, W: int):
        B, d = self._B, self._dmodel
        lid, (_, h, w, c), p3, (_, h3, w3, c3) = self._backbone(camera, B, H, W)

        bev = ttnn.linear(lid, self._bd_w, bias=self._bd_b)  # (1, 1, B*64, d)
        bev = ttnn.reshape(bev, (B, h * w, d))
        st = ttnn.linear(status, self._status[0], bias=self._status[1])  # (B, 1, d)
        keyval = ttnn.add(ttnn.concat([bev, st], dim=1), self._kv_emb)  # (B, 65, d)
        n_kv = h * w + 1

        q = self._tf_decoder(self._query, keyval, B, self._n_query, n_kv)
        context = ttnn.slice(q, [0, 1, 0], [B, self._n_query, d])  # (B, 30, d)

        kv_grid = ttnn.slice(keyval, [0, 0, 0], [B, self._n_kv_grid, d])
        bev_q = ttnn.matmul(self._interp, ttnn.matmul(kv_grid, self._bp_wkv))  # (B, 1024, d)
        p3 = ttnn.reshape(p3, (B, h3 * w3, c3))
        bev_q = ttnn.add(bev_q, ttnn.linear(p3, self._bp_wp3, bias=self._bp_b))
        bev_q = _layer_norm(ttnn.relu(bev_q), self._bp_ln)
        return bev_q, context

    def _meanflow(self, noise, bev_q, context):
        B, K = self._B, self._K
        cond = ttnn.concat(
            [self._cond_time, ttnn.add(bev_q, self._cond_pos_bev), ttnn.add(context, self._cond_pos_ctx)], dim=1
        )
        x = ttnn.add(ttnn.linear(noise, self._mf_in[0], bias=self._mf_in[1]), self._mf_pos)  # (B, K, d)
        x = self._mf_decoder(x, cond, B, K, self._n_cond)
        u = ttnn.linear(_layer_norm(x, self._mf_lnf), self._mf_out[0], bias=self._mf_out[1])
        return ttnn.subtract(noise, u)  # (B, K, T*4)

    def _arm(self, proposals, context):
        B, K, d = self._B, self._K, self._dmodel
        emb = ttnn.linear(ttnn.linear(proposals, self._arm_enc1, activation="relu"), self._arm_enc2)
        dec = self._arm_decoder(emb, context, B, K, context.shape[1])
        emb = ttnn.add(emb, _layer_norm(dec, self._arm_ln))
        y = ttnn.reshape(emb, (B, 1, K * d))
        y = ttnn.silu(ttnn.linear(y, self._arm_recon[0]))
        y = ttnn.silu(ttnn.linear(y, self._arm_recon[1]))
        return ttnn.linear(y, self._arm_recon[2])  # (B, 1, T*4)

    def forward_dev(self, camera, status, noise, H: int, W: int):
        bev_q, context = self._encoder(camera, status, H, W)
        proposals = self._meanflow(noise, bev_q, context)
        return self._arm(proposals, context), proposals

    # -- host boundary -----------------------------------------------------
    def _camera_memory_config(self, n_rows: int):
        grid = ttnn.num_cores_to_corerangeset(CAMERA_CORES, self._d.compute_with_storage_grid_size(), row_wise=True)
        return ttnn.create_sharded_memory_config(
            (n_rows // CAMERA_CORES, CAMERA_CHANNELS),
            grid,
            ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

    def prepare_inputs(self, camera: torch.Tensor, status: torch.Tensor, noise: torch.Tensor, device=None):
        B, C, H, W = camera.shape
        assert B == self._B
        # Writing into a reused zero-padded bf16 buffer is ~10x faster than permute + pad + cast.
        if getattr(self, "_cam_buf", None) is None or self._cam_buf.shape[2] != B * H * W:
            self._cam_buf = torch.zeros(1, 1, B * H * W, CAMERA_CHANNELS, dtype=torch.bfloat16)
        self._cam_buf[0, 0, :, :C].copy_(camera.permute(0, 2, 3, 1).reshape(B * H * W, C))
        cam_t = ttnn.from_torch(
            self._cam_buf,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=self._camera_memory_config(B * H * W) if device is not None else None,
        )
        st_t = ttnn.from_torch(
            status.reshape(B, 1, -1).float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        nz_t = ttnn.from_torch(
            noise.reshape(B, self._K, HORIZON * ACTION_DIM_DELTA).float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        return (cam_t, st_t, nz_t), (H, W)

    def __call__(self, camera: torch.Tensor, status: torch.Tensor, noise: torch.Tensor) -> Dict[str, torch.Tensor]:
        (cam_t, st_t, nz_t), (H, W) = self.prepare_inputs(camera, status, noise, self._d)
        diff_t, prop_t = self.forward_dev(cam_t, st_t, nz_t, H, W)
        return self.postprocess(diff_t, prop_t)

    # -- trace -------------------------------------------------------------
    def capture_trace(self, camera: torch.Tensor, status: torch.Tensor, noise: torch.Tensor) -> None:
        """Warm the program cache and record forward_dev over persistent device inputs."""
        self._trace_in, self._trace_hw = self.prepare_inputs(camera, status, noise, self._d)
        # conv2d uploads and prepares its weights on the first call, so warm twice.
        for _ in range(2):
            self.forward_dev(*self._trace_in, *self._trace_hw)
        ttnn.synchronize_device(self._d)
        self._trace_id = ttnn.begin_trace_capture(self._d, cq_id=0)
        self._trace_out = self.forward_dev(*self._trace_in, *self._trace_hw)
        ttnn.end_trace_capture(self._d, self._trace_id, cq_id=0)

    def run_trace(self, camera: torch.Tensor, status: torch.Tensor, noise: torch.Tensor) -> Dict[str, torch.Tensor]:
        host, _ = self.prepare_inputs(camera, status, noise)
        for h, d in zip(host, self._trace_in):
            ttnn.copy_host_to_device_tensor(h, d)
        ttnn.execute_trace(self._d, self._trace_id, cq_id=0, blocking=False)
        return self.postprocess(*self._trace_out)

    def release_trace(self) -> None:
        if getattr(self, "_trace_id", None) is not None:
            ttnn.release_trace(self._d, self._trace_id)
            self._trace_id = None

    def postprocess(self, diff_t, prop_t) -> Dict[str, torch.Tensor]:
        B, K = self._B, self._K
        diff = ttnn.to_torch(diff_t).float().reshape(B, HORIZON, ACTION_DIM_DELTA)
        props = ttnn.to_torch(prop_t).float().reshape(B, K, HORIZON, ACTION_DIM_DELTA)
        return {
            "trajectory": cumsum_traj(diff),
            "diff_trajectory": diff,
            "pred_diff_traj": props,
        }
