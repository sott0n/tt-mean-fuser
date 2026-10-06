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
    # conv2d needs l1_small; the trace tests need a trace region and a second command queue.
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 << 20, num_command_queues=2)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="session")
def reference_model():
    path = os.environ.get("MF_CHECKPOINT_PATH", str(_DATA_DIR / "meanfuser_navsim.ckpt"))
    if not Path(path).exists():
        pytest.skip(f"MeanFuser checkpoint not found at {path} (set MF_CHECKPOINT_PATH)")
    gmn = os.environ.get("MF_GMN_MEAN_PATH", str(_DATA_DIR / "gmn_center_points.pt"))
    if not Path(gmn).exists():
        pytest.skip(f"GMN center points not found at {gmn} (set MF_GMN_MEAN_PATH)")
    from meanfuser.reference.model import load_model

    return load_model(path, gaussian_mean_path=gmn)


@pytest.fixture
def make_inputs(reference_model):
    """Production-resolution inputs with GMN noise sampled as in upstream inference."""

    def _make(seed: int, batch_size: int = 1):
        g = torch.Generator().manual_seed(seed)
        camera = torch.rand(batch_size, 3, 256, 1024, generator=g)
        command = torch.nn.functional.one_hot(torch.randint(0, 4, (batch_size,), generator=g), 4).float()
        velocity = torch.rand(batch_size, 2, generator=g) * torch.tensor([10.0, 1.0])
        accel = torch.randn(batch_size, 2, generator=g)
        status = torch.cat([command, velocity, accel], dim=1)
        noise = reference_model._meanflow_head.sample_noise(batch_size, generator=g)
        return camera, status, noise

    return _make
