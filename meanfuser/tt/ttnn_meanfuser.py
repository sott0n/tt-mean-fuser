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
    """nn.MultiheadAttention (batch_first, no mask) with explicit q·kᵀ / softmax / ·v."""

    def __init__(self, mha: nn.MultiheadAttention, device) -> None:
        E = mha.embed_dim
        self._nh = mha.num_heads
        self._hd = E // self._nh
        self._E = E
        W, b = mha.in_proj_weight.detach(), mha.in_proj_bias.detach()
        self._wq, self._bq = _tile(W[:E].T, device), _tile(b[:E].reshape(1, -1), device)
        self._wk, self._bk = _tile(W[E : 2 * E].T, device), _tile(b[E : 2 * E].reshape(1, -1), device)
        self._wv, self._bv = _tile(W[2 * E :].T, device), _tile(b[2 * E :].reshape(1, -1), device)
        self._wo, self._bo = _tile(mha.out_proj.weight.T, device), _tile(mha.out_proj.bias.reshape(1, -1), device)
        self._scale = 1.0 / math.sqrt(self._hd)

    def project_kv(self, mem: ttnn.Tensor, B: int, S: int):
        """mem (B, S, E) -> kᵀ (B, nh, hd, S), v (B, nh, S, hd)."""
        k = ttnn.reshape(ttnn.linear(mem, self._wk, bias=self._bk), (B, S, self._nh, self._hd))
        v = ttnn.reshape(ttnn.linear(mem, self._wv, bias=self._bv), (B, S, self._nh, self._hd))
        return ttnn.permute(k, (0, 2, 3, 1)), ttnn.permute(v, (0, 2, 1, 3))

    def __call__(self, x: ttnn.Tensor, B: int, Sq: int, kv) -> ttnn.Tensor:
        k_t, v = kv
        q = ttnn.linear(x, self._wq, bias=self._bq)
        q = ttnn.permute(ttnn.reshape(q, (B, Sq, self._nh, self._hd)), (0, 2, 1, 3))  # (B, nh, Sq, hd)
        att = ttnn.multiply(ttnn.matmul(q, k_t), self._scale)
        att = ttnn.softmax(att, dim=-1, numeric_stable=True)
        o = ttnn.matmul(att, v)  # (B, nh, Sq, hd)
        o = ttnn.reshape(ttnn.permute(o, (0, 2, 1, 3)), (B, Sq, self._E))
        return ttnn.linear(o, self._wo, bias=self._bo)


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

    def __call__(self, x, B: int, Sq: int, mem_kv) -> ttnn.Tensor:
        if self._single:
            sa = ttnn.linear(x, self._sa_w, bias=self._sa_b)
        else:
            sa = self._sa(x, B, Sq, self._sa.project_kv(x, B, Sq))
        x = _layer_norm(ttnn.add(x, sa), self._n1)
        x = _layer_norm(ttnn.add(x, self._ca(x, B, Sq, mem_kv)), self._n2)
        ff = ttnn.linear(x, self._l1[0], bias=self._l1[1], activation="relu")
        ff = ttnn.linear(ff, self._l2[0], bias=self._l2[1])
        return _layer_norm(ttnn.add(x, ff), self._n3)


class _Decoder:
    def __init__(self, decoder: nn.TransformerDecoder, device, single_token_self_attn: bool = False) -> None:
        self._layers = [_DecoderLayer(l, device, single_token_self_attn) for l in decoder.layers]

    def __call__(self, x, memory, B: int, Sq: int, Skv: int) -> ttnn.Tensor:
        for layer in self._layers:
            x = layer(x, B, Sq, layer._ca.project_kv(memory, B, Skv))
        return x


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class _Conv:
    """Plain / BN-folded Conv2d with device-cached weights. NHWC-flat (1,1,B*H*W,C) in and out.

    HiFi4 with fp32 accumulation: at the default fidelity the error compounds through the
    ResNet-34 stages (deepest LiDAR feature PCC ~0.97, trajectory off by ~0.7 m).
    """

    def __init__(self, conv: nn.Conv2d, bn: Optional[nn.BatchNorm2d] = None, stride: Optional[int] = None) -> None:
        if bn is not None:
            w, b = fold_bn(conv, bn)
        else:
            w = conv.weight.detach()
            b = conv.bias.detach() if conv.bias is not None else torch.zeros(conv.out_channels)
        self._w, self._b = prep_conv_weights(w.to(torch.bfloat16), b.to(torch.bfloat16))
        self.cin, self.cout = conv.in_channels, conv.out_channels
        self._k = int(conv.kernel_size[0])
        self._s = int(stride if stride is not None else conv.stride[0])
        self._p = int(conv.padding[0])

    def __call__(self, device, x, B: int, H: int, W: int, relu: bool = True):
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            deallocate_activation=False,
            reallocate_halo_output=True,
            reshard_if_not_optimal=False,
            shard_layout=None,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if relu else None,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
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
            return_weights_and_bias=True,
            return_output_dim=True,
        )
        return _interleaved_tile(out), Ho, Wo


class _BasicBlock:
    """timm ResNet BasicBlock: relu(bn2(conv2(relu(bn1(conv1 x)))) + shortcut(x))."""

    def __init__(self, block, device) -> None:
        self._d = device
        self._c1 = _Conv(block.conv1, block.bn1)
        self._c2 = _Conv(block.conv2, block.bn2)
        self._ds = _Conv(block.downsample[0], block.downsample[1]) if block.downsample is not None else None

    def __call__(self, x, shape):
        B, H, W, _ = shape
        y, H1, W1 = self._c1(self._d, x, B, H, W)
        y, H2, W2 = self._c2(self._d, y, B, H1, W1, relu=False)
        sc = self._ds(self._d, x, B, H, W, relu=False)[0] if self._ds is not None else x
        return ttnn.relu(ttnn.add(y, sc)), (B, H2, W2, self._c2.cout)


class TtnnMeanFuserBackbone:
    """TransFuser backbone with constant LiDAR latent. camera (1,1,B*H*W,3) -> (lidar 8×8×512, FPN 32×32×64)."""

    def __init__(self, ref, device, batch_size: int) -> None:
        self._d = device
        self._B = batch_size
        img = ref.image_encoder
        self._img_stem = _Conv(img.conv1, img.bn1)
        self._img_stages: List[List[_BasicBlock]] = []
        self._lid_stages: List[List[_BasicBlock]] = []
        for i in range(4):
            self._img_stages.append([_BasicBlock(b, device) for b in getattr(img, f"layer{i + 1}")])
            self._lid_stages.append([_BasicBlock(b, device) for b in getattr(ref.lidar_encoder, f"layer{i + 1}")])
        self._fusion = TtnnFuseFeatures(ref, device)

        self._c5 = _Conv(ref.c5_conv)
        self._up5 = _Conv(ref.up_conv5)
        # Only the even pixels of up_conv4's 64×64 output are consumed downstream.
        self._up4 = _Conv(ref.up_conv4, stride=2)
        self._up2 = int(ref.upsample.scale_factor)
        self._up4_size = tuple(ref.upsample2.size)

        self._lid0, self._lid0_shape = self._lidar_layer1(ref)

    def _lidar_layer1(self, ref):
        """Run the input-independent LiDAR stem + layer1 once; the result is reused every frame."""
        lid = ref.lidar_encoder
        with torch.no_grad():
            x = lid.maxpool(lid.act1(lid.bn1(lid.conv1(ref.lidar_latent))))
        x = x.expand(self._B, -1, -1, -1)
        B, C, H, W = x.shape
        t = _tile(x.permute(0, 2, 3, 1).reshape(1, 1, B * H * W, C), self._d)
        shape = (B, H, W, C)
        for blk in self._lid_stages[0]:
            t, shape = blk(t, shape)
        return t, shape

    def _stem(self, x, B: int, H: int, W: int):
        out, Ho, Wo = self._img_stem(self._d, x, B, H, W)
        out = ttnn.max_pool2d(
            ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT),
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
            for blk in self._img_stages[i]:
                img, img_shape = blk(img, img_shape)
            if i > 0:
                for blk in self._lid_stages[i]:
                    lid, lid_shape = blk(lid, lid_shape)
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
        self._tf_decoder = _Decoder(ref._tf_decoder, device)

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
    def prepare_inputs(self, camera: torch.Tensor, status: torch.Tensor, noise: torch.Tensor, device=None):
        B, C, H, W = camera.shape
        assert B == self._B
        cam = camera.permute(0, 2, 3, 1).reshape(1, 1, B * H * W, C)
        cam_t = ttnn.from_torch(cam, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
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
