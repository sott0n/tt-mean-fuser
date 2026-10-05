# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MeanFuser reference model — inference-only PyTorch port, no nuplan/navsim dependencies.

  camera (B×3×256×1024) + ego status (B×8)
  -> TransFuser backbone (ResNet-34 image + ResNet-34 on a learned LiDAR latent, 4-scale GPT fusion)
  -> perception TransformerDecoder (3 layers, 65 key-val tokens) -> context query (B×30×128)
  -> bev query (B×1024×128) from the FPN output and the 8×8 key-val grid
  -> MeanFlow head: one-step mean-velocity prediction for K=8 GMN noise samples
  -> ARM: 3-layer cross-attention over the K proposals -> final trajectory (B×8×3)

Upstream: https://github.com/wjl2244/MeanFuser (navsim/agents/meanfuser)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental.diffusion_drive.reference.model import DiffusionDriveConfig, TransfuserBackbone

HORIZON = 8
ACTION_DIM_DELTA = 4
ACTION_DIM_ORI = 3

# Normalisation constants of diff_traj / cumsum_traj, fixed by the upstream training set.
X_DIFF_MIN = -1.2698211669921875
X_DIFF_MAX = 7.475563049316406
X_DIFF_MEAN = 2.950225591659546
Y_DIFF_MIN = -5.012081146240234
Y_DIFF_MAX = 4.8563690185546875
Y_DIFF_MEAN = 0.0607292577624321
X_DIFF_RANGE = max(abs(X_DIFF_MAX - X_DIFF_MEAN), abs(X_DIFF_MIN - X_DIFF_MEAN))
Y_DIFF_RANGE = max(abs(Y_DIFF_MAX - Y_DIFF_MEAN), abs(Y_DIFF_MIN - Y_DIFF_MEAN))


@dataclass
class MeanFuserConfig:
    camera_width: int = 1024
    camera_height: int = 256
    lidar_resolution_width: int = 256
    lidar_resolution_height: int = 256
    # The released checkpoint was trained with lidar_seq_len=4, so the latent has 4 channels.
    lidar_seq_len: int = 4

    tf_d_model: int = 128
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    num_bounding_boxes: int = 30

    num_proposals: int = 8
    decoder_layers: int = 1
    decoder_nhead: int = 8
    arm_layers: int = 3

    bev_features_channels: int = 64
    num_bev_classes: int = 7

    def backbone_config(self) -> DiffusionDriveConfig:
        return DiffusionDriveConfig(
            latent=True,
            camera_width=self.camera_width,
            camera_height=self.camera_height,
            lidar_resolution_width=self.lidar_resolution_width,
            lidar_resolution_height=self.lidar_resolution_height,
            lidar_seq_len=self.lidar_seq_len,
            img_vert_anchors=self.camera_height // 32,
            img_horz_anchors=self.camera_width // 32,
            lidar_vert_anchors=self.lidar_resolution_height // 32,
            lidar_horz_anchors=self.lidar_resolution_width // 32,
            bev_features_channels=self.bev_features_channels,
        )


def cumsum_traj(norm_trajs: torch.Tensor) -> torch.Tensor:
    """(B, T, 4) normalised (dx, dy, sin, cos) -> (B, T, 3) absolute (x, y, heading)."""
    heading = torch.atan2(norm_trajs[..., 2:3], norm_trajs[..., 3:4])
    x = (norm_trajs[..., 0:1] * X_DIFF_RANGE + X_DIFF_MEAN).cumsum(dim=1)
    y = (norm_trajs[..., 1:2] * Y_DIFF_RANGE + Y_DIFF_MEAN).cumsum(dim=1)
    return torch.cat([x, y, heading], -1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def _decoder(d_model: int, nhead: int, d_ffn: int, num_layers: int) -> nn.TransformerDecoder:
    layer = nn.TransformerDecoderLayer(d_model, nhead, d_ffn, dropout=0.0, batch_first=True)
    return nn.TransformerDecoder(layer, num_layers)


class SimpleDiffusionTransformer(nn.Module):
    """Mean-velocity network u(z, r, t | cond). One query token per proposal."""

    def __init__(self, d_model: int, nhead: int, d_ffn: int, num_layers: int, input_dim: int, obs_len: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_decoder(d_model, nhead, d_ffn, num_layers)])
        self.input_emb = nn.Linear(input_dim, d_model)
        self.time_t_emb = SinusoidalPosEmb(d_model)
        self.time_r_emb = SinusoidalPosEmb(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.output_emb = nn.Linear(d_model, input_dim)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, obs_len + 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, d_model))

    def time_embedding(self, time_r: torch.Tensor, time_t: torch.Tensor) -> torch.Tensor:
        # The r branch embeds the interval t - r, not r itself.
        return (self.time_t_emb(time_t) + self.time_r_emb(time_t - time_r)).unsqueeze(1)

    def forward(
        self,
        sample: torch.Tensor,
        time_r: torch.Tensor,
        time_t: torch.Tensor,
        bev_query: torch.Tensor,
        context_query: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = sample.shape
        x = self.input_emb(sample.reshape(B, -1).float()).unsqueeze(1) + self.pos_emb
        cond = torch.cat([self.time_embedding(time_r, time_t), bev_query, context_query], dim=1)
        cond = cond + self.cond_pos_emb[:, : cond.shape[1], :]
        for block in self.blocks:
            x = block(tgt=x, memory=cond)
        x = self.output_emb(self.ln_f(x))
        return x.squeeze(1).view(B, T, D)


class MeanFlowHead(nn.Module):
    def __init__(self, config: MeanFuserConfig) -> None:
        super().__init__()
        self.config = config
        obs_len = (config.lidar_resolution_height // 8) * (
            config.lidar_resolution_width // 8
        ) + config.num_bounding_boxes
        self.model = SimpleDiffusionTransformer(
            config.tf_d_model,
            config.decoder_nhead,
            config.tf_d_ffn,
            config.decoder_layers,
            input_dim=ACTION_DIM_DELTA * HORIZON,
            obs_len=obs_len,
        )
        K = config.num_proposals
        self.cluster_trajs = nn.Parameter(torch.zeros(K, HORIZON, ACTION_DIM_ORI), requires_grad=False)
        self.gaussian_std = nn.Parameter(torch.ones(K, ACTION_DIM_DELTA), requires_grad=False)
        # Upstream keeps the GMN means outside the state_dict (loaded from navtrain_8_mean_std.pkl,
        # scaled by 0.5); set them with set_gaussian_mean().
        self.register_buffer("gaussian_mean", torch.zeros(K, ACTION_DIM_DELTA), persistent=False)

    def set_gaussian_mean(self, center_points: torch.Tensor) -> None:
        self.gaussian_mean.copy_(center_points.float() * 0.5)

    def sample_noise(self, batch_size: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """GMN noise in normalised diff space, (B, K, T, 4)."""
        K = self.config.num_proposals
        mean = self.gaussian_mean[None, :, None, :]
        std = self.gaussian_std[None, :, None, :]
        e = torch.randn((batch_size, K, HORIZON, ACTION_DIM_DELTA), generator=generator, device=mean.device)
        return e * std + mean

    def forward(self, bev_query: torch.Tensor, context_query: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """One-step sample: x0 = e - u(e, r=0, t=1). Returns (B, K, T, 4)."""
        B, K = noise.shape[:2]
        e = noise.reshape(B * K, HORIZON, ACTION_DIM_DELTA)
        r = torch.zeros(B * K, device=e.device)
        t = torch.ones(B * K, device=e.device)
        u = self.model(
            e,
            r,
            t,
            bev_query.repeat_interleave(K, dim=0),
            context_query.repeat_interleave(K, dim=0),
        )
        return (e - u).view(B, K, HORIZON, ACTION_DIM_DELTA)


class SemanticMapHead(nn.Module):
    """Training-only auxiliary head; kept so the checkpoint loads strictly."""

    def __init__(self, config: MeanFuserConfig) -> None:
        super().__init__()
        c = config.bev_features_channels
        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, config.num_bev_classes, kernel_size=1),
        )


class ARMModel(nn.Module):
    """Adaptive Reconstruction Module: fuses the K proposals into one trajectory."""

    def __init__(self, config: MeanFuserConfig) -> None:
        super().__init__()
        d = config.tf_d_model
        self.traj_encoder = nn.Sequential(
            nn.Linear(HORIZON * ACTION_DIM_DELTA, d * 2, bias=False),
            nn.ReLU(),
            nn.Linear(d * 2, d, bias=False),
        )
        self.bev_cross_attn = _decoder(d, config.tf_num_head, config.tf_d_ffn, config.arm_layers)
        self.norm1 = nn.LayerNorm(d)
        self.trajectory_recon = nn.Sequential(
            nn.Linear(config.num_proposals * d, d, bias=False),
            nn.SiLU(),
            nn.Linear(d, d, bias=False),
            nn.SiLU(),
            nn.Linear(d, HORIZON * ACTION_DIM_DELTA, bias=False),
        )

    def forward(self, proposals: torch.Tensor, context_query: torch.Tensor) -> torch.Tensor:
        """(B, K, T, 4) proposals -> (B, T, 4) normalised diff trajectory."""
        B, K = proposals.shape[:2]
        emb = self.traj_encoder(proposals.reshape(B, K, -1))
        emb = emb + self.norm1(self.bev_cross_attn(emb, context_query))
        return self.trajectory_recon(emb.reshape(B, -1)).view(B, HORIZON, ACTION_DIM_DELTA)


class MeanFuserModel(nn.Module):
    def __init__(self, config: MeanFuserConfig) -> None:
        super().__init__()
        self.config = config
        d = config.tf_d_model
        bb_cfg = config.backbone_config()
        self._backbone = TransfuserBackbone(bb_cfg)
        self._status_encoding = nn.Linear(4 + 2 + 2, d)
        self._bev_downscale = nn.Conv2d(512, d, kernel_size=1)
        self.bev_proj = nn.Sequential(
            nn.Linear(d + config.bev_features_channels, d),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d),
        )
        self._query_splits = [1, config.num_bounding_boxes]
        self._query_embedding = nn.Embedding(config.num_bounding_boxes + 1, d)
        n_kv = bb_cfg.lidar_vert_anchors * bb_cfg.lidar_horz_anchors + 1
        self._keyval_embedding = nn.Embedding(n_kv, d)
        self._tf_decoder = _decoder(d, config.tf_num_head, config.tf_d_ffn, config.tf_num_layers)

        self._meanflow_head = MeanFlowHead(config)
        self._semantic_map_head = SemanticMapHead(config)
        self._arm_model_head = ARMModel(config)

    def encoder(self, camera: torch.Tensor, status: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = status.shape[0]
        bev_upscale, bev_feature, _ = self._backbone(camera, None)
        bev_hw = bev_upscale.shape[2:]
        kv_hw = bev_feature.shape[2:]

        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1).permute(0, 2, 1)
        status_encoding = self._status_encoding(status).unsqueeze(1)
        keyval = torch.cat([bev_feature, status_encoding], dim=1) + self._keyval_embedding.weight[None]

        kv_grid = keyval[:, :-1].permute(0, 2, 1).reshape(B, -1, kv_hw[0], kv_hw[1])
        kv_grid = F.interpolate(kv_grid, size=bev_hw, mode="bilinear", align_corners=False)
        cross_bev = torch.cat([kv_grid, bev_upscale], dim=1)
        cross_bev = F.interpolate(cross_bev, scale_factor=0.5)
        bev_query = self.bev_proj(cross_bev.flatten(-2, -1).permute(0, 2, 1))

        query = self._query_embedding.weight[None].repeat(B, 1, 1)
        query_out = self._tf_decoder(query, keyval)
        _, context_query = query_out.split(self._query_splits, dim=1)
        return {"bev_query": bev_query, "context_query": context_query, "bev_feature_upscale": bev_upscale}

    def forward(
        self,
        camera: torch.Tensor,
        status: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            camera: (B, 3, 256, 1024) stitched front camera, [0, 1].
            status: (B, 8) driving command (4) + ego velocity (2) + ego acceleration (2).
            noise:  optional (B, K, 8, 4) GMN noise; sampled from the head when None.
        """
        enc = self.encoder(camera, status)
        if noise is None:
            noise = self._meanflow_head.sample_noise(status.shape[0], generator)
        proposals = self._meanflow_head(enc["bev_query"], enc["context_query"], noise)
        diff_traj = self._arm_model_head(proposals, enc["context_query"])
        B, K = proposals.shape[:2]
        return {
            "trajectory": cumsum_traj(diff_traj),
            "diff_trajectory": diff_traj,
            "pred_diff_traj": proposals,
            "pred_trajectorys": cumsum_traj(proposals.reshape(B * K, HORIZON, -1)).view(B, K, HORIZON, -1),
            **enc,
        }


def load_model(
    checkpoint_path: str,
    config: Optional[MeanFuserConfig] = None,
    gaussian_mean_path: Optional[str] = None,
) -> MeanFuserModel:
    """Load the released Lightning checkpoint strictly. ``gaussian_mean_path`` is a
    tensor file holding the (K, 4) GMN ``center_points``."""
    model = MeanFuserModel(config or MeanFuserConfig())
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)
    prefix = "agent.meanfuser_model."
    state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    model.load_state_dict(state_dict, strict=True)
    if gaussian_mean_path is not None:
        model._meanflow_head.set_gaussian_mean(torch.load(gaussian_mean_path, weights_only=True))
    return model.eval()
