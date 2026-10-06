# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from meanfuser.tt.ttnn_meanfuser import TtnnMeanFuser

TRAJ_PCC = 0.9999
# Waypoints reach ~20 m at 4 s; bf16 activations leave errors of ~0.1 m at the far end.
TRAJ_MAX_ABS_ERR_M = 0.25


def _reference(ref, camera, status, noise):
    with torch.no_grad():
        return ref(camera, status, noise=noise)


def _check_trajectory(expected, actual):
    passed, pcc = comp_pcc(expected, actual, TRAJ_PCC)
    err = (expected - actual).abs().max().item()
    assert passed, f"trajectory PCC {pcc}"
    assert err < TRAJ_MAX_ABS_ERR_M, f"trajectory max abs error {err:.3f} m"


@pytest.mark.timeout(600)
def test_eager_matches_reference(device, reference_model, make_inputs):
    camera, status, noise = make_inputs(seed=0)
    expected = _reference(reference_model, camera, status, noise)

    tt = TtnnMeanFuser(reference_model, device, batch_size=1)
    (cam_t, st_t, nz_t), (H, W) = tt.prepare_inputs(camera, status, noise, device)
    bev_q, context = tt._encoder(cam_t, st_t, H, W)
    for name, exp, act in (
        ("bev_query", expected["bev_query"], bev_q),
        ("context_query", expected["context_query"], context),
    ):
        passed, pcc = comp_pcc(exp, ttnn.to_torch(act).float(), 0.999)
        assert passed, f"{name} PCC {pcc}"

    actual = tt(camera, status, noise)
    passed, pcc = comp_pcc(expected["pred_diff_traj"], actual["pred_diff_traj"], 0.999)
    assert passed, f"proposals PCC {pcc}"
    _check_trajectory(expected["trajectory"], actual["trajectory"])


@pytest.mark.timeout(600)
def test_trace_matches_reference(device, reference_model, make_inputs):
    """Capture on one input and replay on others: a replay that ignored the input
    refresh would reproduce the capture-time trajectory instead of the new one."""
    tt = TtnnMeanFuser(reference_model, device, batch_size=1)
    capture_inputs = make_inputs(seed=100)
    tt.capture_trace(*capture_inputs)
    try:
        stale = _reference(reference_model, *capture_inputs)["trajectory"]
        for seed in (1, 2):
            inputs = make_inputs(seed=seed)
            expected = _reference(reference_model, *inputs)["trajectory"]
            actual = tt.run_trace(*inputs)["trajectory"]
            _check_trajectory(expected, actual)
            assert (stale - actual).abs().max() > 4 * TRAJ_MAX_ABS_ERR_M, "trace replay ignored the new inputs"
    finally:
        tt.release_trace()


@pytest.mark.timeout(600)
def test_trace_2cq_matches_reference(device, reference_model, make_inputs):
    """Pipelined 2CQ replay: every yielded result must belong to its own frame, in order."""
    tt = TtnnMeanFuser(reference_model, device, batch_size=1)
    tt.capture_trace_2cq(*make_inputs(seed=100))
    frames = [make_inputs(seed=s) for s in (1, 2, 3, 4, 5)]
    try:
        results = list(tt.run_trace_2cq(frames))
        assert len(results) == len(frames)
        for frame, result in zip(frames, results):
            _check_trajectory(_reference(reference_model, *frame)["trajectory"], result["trajectory"])
    finally:
        tt.release_trace()
