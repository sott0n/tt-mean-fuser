# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""HTTP server for MeanFuser on one Blackhole chip (tt-model `tt-dit-server` kind).

POST /v1/plan    {"image": <base64 PNG/JPEG, 1024x256 RGB, stitched L0/F0/R0>,
                  "status": [8 floats: driving command one-hot (4), velocity xy, acceleration xy],
                  "seed": int | null}
                 -> {"trajectory": [[x, y, heading] x 8], "timing_ms": {...}}
GET  /v1/health  -> {"status": "ok" | "loading" | "error", ...}
GET  /v1/models  -> the served model id

Weights are not bundled. MF_CHECKPOINT_PATH and MF_GMN_MEAN_PATH name the upstream files.
The device opens and the trace is captured at startup, so "Application startup complete"
means ready. Requests are serialised on the chip.
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

MODEL_ID = "meanfuser"
IMAGE_SIZE = (1024, 256)  # (width, height)
STATUS_DIM = 8

STATE = {"status": "loading", "error": None, "ref": None, "tt": None, "device": None}
LOCK = threading.Lock()


def _required_path(var: str) -> str:
    path = os.environ.get(var)
    if not path or not os.path.isfile(path):
        raise RuntimeError(f"{var} must name an existing file (got {path!r})")
    return path


def _default_mesh_descriptor() -> None:
    # One chip of a 2-chip p300 is detected as a CUSTOM cluster and needs a 1x1 mesh descriptor.
    # The server always runs on one chip, so the P150 descriptor fits every board. Its path is
    # inside the installed ttnn, which a user of a pulled bundle cannot easily name.
    if "TT_MESH_GRAPH_DESC_PATH" in os.environ:
        return
    import importlib.util

    ttnn_dir = os.path.dirname(importlib.util.find_spec("ttnn").origin)
    desc = os.path.join(ttnn_dir, "tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto")
    if os.path.isfile(desc):
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = desc


def _load():
    _default_mesh_descriptor()
    import ttnn

    from meanfuser.reference.model import load_model
    from meanfuser.tt.ttnn_meanfuser import TtnnMeanFuser

    ref = load_model(_required_path("MF_CHECKPOINT_PATH"), gaussian_mean_path=_required_path("MF_GMN_MEAN_PATH"))
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 << 20)
    tt = TtnnMeanFuser(ref, device, batch_size=1)
    # Every request reuses this trace, so capture it once with inputs of the served shape.
    camera = torch.zeros(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0])
    status = torch.zeros(1, STATUS_DIM)
    tt.capture_trace(camera, status, ref._meanflow_head.sample_noise(1, generator=torch.Generator().manual_seed(0)))
    STATE.update(ref=ref, tt=tt, device=device, status="ok")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        _load()
    except Exception as e:
        STATE.update(status="error", error=f"{type(e).__name__}: {e}")
        raise
    yield
    import ttnn

    with LOCK:
        if STATE["tt"] is not None:
            STATE["tt"].release_trace()
        if STATE["device"] is not None:
            ttnn.close_device(STATE["device"])


app = FastAPI(title="MeanFuser", lifespan=lifespan)


class PlanRequest(BaseModel):
    image: str = Field(description="base64 PNG/JPEG, 1024x256 RGB, stitched L0/F0/R0")
    status: List[float] = Field(min_length=STATUS_DIM, max_length=STATUS_DIM)
    seed: Optional[int] = None


class PlanResponse(BaseModel):
    trajectory: List[List[float]]
    timing_ms: dict


def _decode_image(b64: str) -> torch.Tensor:
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64, validate=True))).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"image is not a base64 PNG/JPEG: {e}")
    if img.size != IMAGE_SIZE:
        raise HTTPException(status_code=422, detail=f"image must be {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}, got {img.size}")
    return torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(255.0)[None]


@app.post("/v1/plan", response_model=PlanResponse)
def plan(req: PlanRequest) -> PlanResponse:
    if STATE["status"] != "ok":
        raise HTTPException(status_code=503, detail=f"model {STATE['status']}")
    t0 = time.perf_counter()
    camera = _decode_image(req.image)
    status = torch.tensor([req.status], dtype=torch.float32)
    generator = torch.Generator().manual_seed(req.seed) if req.seed is not None else None
    noise = STATE["ref"]._meanflow_head.sample_noise(1, generator=generator)
    t1 = time.perf_counter()
    with LOCK:
        trajectory = STATE["tt"].run_trace(camera, status, noise)["trajectory"][0]
    t2 = time.perf_counter()
    return PlanResponse(
        trajectory=trajectory.tolist(),
        timing_ms={"preprocess": (t1 - t0) * 1e3, "device": (t2 - t1) * 1e3},
    )


@app.get("/v1/health")
def health() -> dict:
    return {"status": STATE["status"], "error": STATE["error"]}


@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "tenstorrent"}]}
