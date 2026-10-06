# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NAVSIM v1 PDM score of MeanFuser on navtest, for the TTNN model and/or the CPU reference.

  infer:  navtest scenes -> <out>/trajectories_<backend>.npz  (token -> (8, 3) x, y, heading)
  score:  trajectories + metric cache -> <out>/pdm_<backend>.csv

Both backends see the same per-token GMN noise, so their scores differ only by numerics.

Runs in a Python env that has the navsim devkit (upstream MeanFuser repo) and nuplan-devkit
next to ttnn. Expects the standard NAVSIM variables: OPENSCENE_DATA_ROOT, NUPLAN_MAPS_ROOT,
NUPLAN_MAP_VERSION, NAVSIM_EXP_ROOT, NAVSIM_CACHE_ROOT. The metric cache comes from the devkit's
run_metric_caching.py.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

BACKENDS = ("ttnn", "torch")


def _compose(overrides: List[str]):
    from hydra import compose, initialize_config_module

    with initialize_config_module(config_module="navsim.planning.script.config.pdm_scoring", version_base=None):
        return compose(config_name="default_run_pdm_score", overrides=overrides)


def _features(agent_input):
    """Upstream MFFeatureBuilder camera stitching and ego status, without the unused LiDAR feature."""
    cam = agent_input.cameras[-1]
    l0 = cam.cam_l0.image[28:-28, 416:-416]
    f0 = cam.cam_f0.image[28:-28]
    r0 = cam.cam_r0.image[28:-28, 416:-416]
    image = cv2.resize(np.concatenate([l0, f0, r0], axis=1), (1024, 256))
    camera = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)[None]
    s = agent_input.ego_statuses[-1]
    status = torch.cat(
        [
            torch.as_tensor(s.driving_command, dtype=torch.float32),
            torch.as_tensor(s.ego_velocity, dtype=torch.float32),
            torch.as_tensor(s.ego_acceleration, dtype=torch.float32),
        ]
    )[None]
    return camera, status


def _token_generator(token: str) -> torch.Generator:
    return torch.Generator().manual_seed(int(hashlib.sha256(token.encode()).hexdigest()[:15], 16))


def infer(cfg, args) -> None:
    from hydra.utils import instantiate

    from models.experimental.meanfuser.reference.model import load_model
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader

    sensors = SensorConfig(
        cam_f0=[3],
        cam_l0=[3],
        cam_l1=[],
        cam_l2=[],
        cam_r0=[3],
        cam_r1=[],
        cam_r2=[],
        cam_b0=[],
        lidar_pc=[],
    )
    loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=sensors,
    )
    tokens = sorted(loader.tokens)[: args.limit or None]
    ref = load_model(args.checkpoint, gaussian_mean_path=args.gmn_mean)

    tt, device = None, None
    if "ttnn" in args.backends:
        import ttnn

        from models.experimental.meanfuser.tt.ttnn_meanfuser import TtnnMeanFuser

        device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 << 20)
        tt = TtnnMeanFuser(ref, device, batch_size=1)

    results: Dict[str, Dict[str, np.ndarray]] = {b: {} for b in args.backends}
    t0 = time.time()
    try:
        for i, token in enumerate(tokens):
            camera, status = _features(loader.get_agent_input_from_token(token))
            noise = ref._meanflow_head.sample_noise(1, generator=_token_generator(token))
            if tt is not None:
                if i == 0:
                    tt.capture_trace(camera, status, noise)
                results["ttnn"][token] = tt.run_trace(camera, status, noise)["trajectory"][0].numpy()
            if "torch" in args.backends:
                with torch.no_grad():
                    results["torch"][token] = ref(camera, status, noise=noise)["trajectory"][0].numpy()
            if (i + 1) % 500 == 0:
                print(
                    f"[infer] {i + 1}/{len(tokens)} scenes, {(time.time() - t0) / (i + 1) * 1e3:.1f} ms/scene",
                    flush=True,
                )
    finally:
        if tt is not None:
            tt.release_trace()
            ttnn.close_device(device)

    for backend, trajs in results.items():
        path = Path(args.out) / f"trajectories_{backend}.npz"
        np.savez(path, **{k: v.astype(np.float32) for k, v in trajs.items()})
        print(f"[infer] wrote {len(trajs)} trajectories to {path}")


def score(cfg, args) -> None:
    import pandas as pd
    from hydra.utils import instantiate

    from navsim.common.dataclasses import SensorConfig, Trajectory
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.script.builders.worker_pool_builder import build_worker
    from navsim.planning.script.run_pdm_score_gpu_v1 import run_pdm_score
    from nuplan.planning.utils.multithreading.worker_utils import worker_map

    loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        sensor_blobs_path=None,
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    worker = build_worker(cfg)
    for backend in args.backends:
        data = np.load(Path(args.out) / f"trajectories_{backend}.npz")
        trajectories = {t: {"trajectory": Trajectory(data[t])} for t in data.files}
        data_points = [
            {
                "cfg": cfg,
                "log_file": log,
                "tokens": [t for t in toks if t in trajectories],
                "model_trajectory": trajectories,
            }
            for log, toks in loader.get_tokens_list_per_log().items()
        ]
        data_points = [d for d in data_points if d["tokens"]]
        df = pd.DataFrame(worker_map(worker, run_pdm_score, data_points))
        valid = df[df["valid"]]
        csv = Path(args.out) / f"pdm_{backend}.csv"
        df.to_csv(csv, index=False)
        metrics = valid.drop(columns=["token", "valid"]).mean()
        print(f"[score] {backend}: {len(valid)}/{len(df)} valid scenes, PDMS {metrics['score']:.4f} -> {csv}")
        print(metrics.to_string())


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("infer", "score", "all"))
    p.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    p.add_argument("--out", required=True)
    p.add_argument(
        "--checkpoint", default=os.environ.get("MF_CHECKPOINT_PATH", str(data_dir / "meanfuser_navsim.ckpt"))
    )
    p.add_argument("--gmn-mean", default=os.environ.get("MF_GMN_MEAN_PATH", str(data_dir / "gmn_center_points.pt")))
    p.add_argument("--split", default="navtest")
    p.add_argument("--limit", type=int, default=0, help="first N scenes only (0 = all)")
    p.add_argument("--worker", default="ray_distributed_no_torch")
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    cfg = _compose(
        [
            f"train_test_split={args.split}",
            f"worker={args.worker}",
            "experiment_name=meanfuser_pdm",
            f"metric_cache_path={os.environ['NAVSIM_CACHE_ROOT']}/{args.split}_v1_metric_cache",
        ]
    )
    if args.mode in ("infer", "all"):
        infer(cfg, args)
    if args.mode in ("score", "all"):
        score(cfg, args)


if __name__ == "__main__":
    main()
