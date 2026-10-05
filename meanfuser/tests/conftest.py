# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import ttnn

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session")
def device():
    # conv2d needs l1_small; the trace test needs a trace region.
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 << 20)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="session")
def reference_model():
    path = os.environ.get("MF_CHECKPOINT_PATH", str(_DATA_DIR / "meanfuser_navsim.ckpt"))
    if not Path(path).exists():
        pytest.skip(f"MeanFuser checkpoint not found at {path} (set MF_CHECKPOINT_PATH)")
    from models.experimental.meanfuser.reference.model import load_model

    return load_model(path)


@pytest.fixture
def make_inputs(reference_model):
    """Production-resolution inputs; GMN noise uses the checkpoint's per-mode std."""

    def _make(seed: int, batch_size: int = 1):
        g = torch.Generator().manual_seed(seed)
        camera = torch.rand(batch_size, 3, 256, 1024, generator=g)
        command = torch.nn.functional.one_hot(torch.randint(0, 4, (batch_size,), generator=g), 4).float()
        velocity = torch.rand(batch_size, 2, generator=g) * torch.tensor([10.0, 1.0])
        accel = torch.randn(batch_size, 2, generator=g)
        status = torch.cat([command, velocity, accel], dim=1)
        std = reference_model._meanflow_head.gaussian_std
        noise = torch.randn(batch_size, std.shape[0], 8, 4, generator=g) * std[None, :, None, :]
        return camera, status, noise

    return _make
