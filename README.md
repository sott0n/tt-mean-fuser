# MeanFuser on Tenstorrent Blackhole (TTNN)

TTNN port of [MeanFuser](https://github.com/wjl2244/MeanFuser) (CVPR 2026), a one-step
MeanFlow end-to-end driving planner for NAVSIM. It runs on one Blackhole chip.

| Metric (1 BH chip, batch 1) | Value |
|---|---|
| navtest PDMS, 12146 scenes | TTNN 0.8911, CPU fp32 reference 0.8903, paper 89.0 |
| Trace replay (device) | 4.62 ms |
| Single-frame latency, 1 CQ trace | 5.7-6.3 ms (host-side variance) |
| Throughput, 2 CQ pipelined trace | 4.86 ms/frame (~206 FPS) |
| Trajectory vs CPU reference | mean 0.096 m, p99 0.39 m (max abs xy error) |

## Layout

```
meanfuser/
  reference/model.py        PyTorch reference; matches upstream bit-exactly
  tt/ttnn_meanfuser.py      TTNN model, 1 CQ and 2 CQ trace paths
  tests/pcc/                PCC tests: eager, trace, 2 CQ trace
  server/app.py             HTTP server, served by tt-model
  scripts/navsim_pdm_eval.py  navtest PDMS for TTNN and/or the CPU reference
  scripts/package_tt_model.sh  stage the tt-model bundle
third_party/tt-metal        tt-metal submodule (pinned)
```

The model reuses the ResNet-34/GPT-fusion pieces of tt-metal's
`models/experimental/diffusion_drive` and `models/tt_cnn`, so tt-metal must be on `PYTHONPATH`.

## Setup

```bash
git clone --recursive git@github.com:sott0n/tt-mean-fuser.git
cd tt-mean-fuser/third_party/tt-metal
./build_metal.sh && ./create_venv.sh     # see tt-metal docs
cd ../..
source third_party/tt-metal/python_env/bin/activate
export TT_METAL_HOME=$PWD/third_party/tt-metal
export PYTHONPATH=$PWD:$TT_METAL_HOME
```

### Selecting one chip

```bash
export TT_VISIBLE_DEVICES=3   # /dev/tenstorrent/3, seen as device 0 in-process
# One chip of a 2-chip p300 board is detected as a CUSTOM cluster; give it a 1x1 mesh:
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
```

### Assets

Put these in `meanfuser/data/` (gitignored; a symlink works), or point `MF_CHECKPOINT_PATH` /
`MF_GMN_MEAN_PATH` at them.

| File | Source |
|---|---|
| `meanfuser_navsim.ckpt` | [upstream checkpoint](https://drive.google.com/file/d/16989kIYhM3wQgxjSKvRFfK9cdKZfuU2P/view) (PDMS 89.0, Google Drive) |
| `gmn_center_points.pt` | `center_points` of upstream [`navtrain_8_mean_std.pkl`](https://github.com/wjl2244/MeanFuser/blob/8de8ba6244834645192e318dcc437d124cfd6872/tools/gaussian_mixed_noise/navtrain_8_mean_std.pkl) |

The `.pkl` is a full pickle (protocol 4); convert it once, from a source you trust:

```bash
python -c "import pickle,torch; m=pickle.load(open('navtrain_8_mean_std.pkl','rb')); \
torch.save(torch.as_tensor(m['center_points']).float(), 'gmn_center_points.pt')"
```

The checkpoint itself loads with `torch.load(weights_only=True)`.

## Tests

```bash
pytest meanfuser/tests
```

## Usage

```python
import ttnn
from meanfuser.reference.model import load_model
from meanfuser.tt.ttnn_meanfuser import TtnnMeanFuser

ref = load_model("meanfuser/data/meanfuser_navsim.ckpt", gaussian_mean_path="meanfuser/data/gmn_center_points.pt")
dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 << 20, num_command_queues=2)
tt = TtnnMeanFuser(ref, dev, batch_size=1)

# camera: (1, 3, 256, 1024) stitched L0/F0/R0 in [0, 1]; status: (1, 8); noise: (1, 8, 8, 4)
noise = ref._meanflow_head.sample_noise(1)
tt.capture_trace(camera, status, noise)          # 1 CQ, lowest latency
out = tt.run_trace(camera, status, noise)["trajectory"]   # (1, 8, 3) x, y, heading

tt.capture_trace_2cq(camera, status, noise)      # 2 CQ, highest throughput
for out in tt.run_trace_2cq(frames):             # frames: iterable of (camera, status, noise)
    ...
```

## Serving with tt-model

`meanfuser/server/app.py` is an HTTP server for the model. [tt-model](https://github.com/tenstorrent/tt-model-manager)
serves it as a v6 thin bundle with `kind: tt-dit-server` (uvicorn, no vLLM).

```
POST /v1/plan    {"image": <base64 PNG/JPEG, 1024x256 RGB, stitched L0/F0/R0>,
                  "status": [command one-hot (4), velocity xy, acceleration xy], "seed": int | null}
                 -> {"trajectory": [[x, y, heading] x 8], "timing_ms": {...}}
GET  /v1/health, GET /v1/models
```

The bundle carries no weights. Get the two files in [Assets](#assets) from upstream, then:

```bash
MF_CHECKPOINT_PATH=/path/to/meanfuser_navsim.ckpt MF_GMN_MEAN_PATH=/path/to/gmn_center_points.pt \
TT_VISIBLE_DEVICES=3 tt-model serve <org>/meanfuser --port 8000
```

- `--port` picks the port (default 20000).
- The host needs SFPI 7.84.0 in `/opt/tenstorrent/sfpi`. ttnn wheels do not bundle SFPI.
- The first start compiles kernels, which takes ~10 min. Later starts reuse the cache.
- Ready when uvicorn prints "Application startup complete": the trace is captured at startup.

To build the bundle from this checkout (needs a built tt-metal, `uv` and `tt-model`):

```bash
meanfuser/scripts/package_tt_model.sh <out-dir>                  # stage only
meanfuser/scripts/package_tt_model.sh <out-dir> <org>/meanfuser  # stage and push
```

It bundles a ttnn wheel of the pinned tt-metal, since no index has one, and a `meanfuser`
wheel that also ships tt-metal's `diffusion_drive` modules.

## navtest PDMS

The eval runs in a separate venv with the NAVSIM devkit (the upstream MeanFuser repo) and
nuplan-devkit. It takes ttnn and torch from the tt-metal checkout in `$TT_METAL_HOME`. Build it
once (needs `uv`):

```bash
meanfuser/scripts/setup_navsim_venv.sh <navsim-root>   # devkits + venv under <navsim-root>
```

Then, for each run, set the tt-metal side and activate the venv:

```bash
export TT_METAL_HOME=$PWD/third_party/tt-metal PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=3 TT_MESH_GRAPH_DESC_PATH=...   # as in "Selecting one chip"
export NAVSIM_DEVKIT_ROOT=<navsim-root>/MeanFuser OPENSCENE_DATA_ROOT=... NUPLAN_MAPS_ROOT=...
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NAVSIM_EXP_ROOT=... NAVSIM_CACHE_ROOT=...
source <navsim-root>/venv/bin/activate
```

1. Download the navtest split. Only `CAM_F0`, `CAM_L0` and `CAM_R0` are needed (~45 GB after
   extraction). Also download `openscene_metadata_test` and the nuPlan maps.
2. Build the metric cache:
   `python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py train_test_split=navtest cache.cache_path=$NAVSIM_CACHE_ROOT/navtest_v1_metric_cache`
3. Run inference and scoring:
   `python meanfuser/scripts/navsim_pdm_eval.py all --out <dir>` (add `--backends ttnn` for TTNN only)

   `infer` needs the device. `score` is CPU only. They can run as separate steps on the same
   `--out`. TTNN `infer` on navtest takes ~22 ms/scene, ~4.5 min in total.

Both backends use the same per-token GMN noise, so their scores differ only by numerics.

## Notes

- Convs run at HiFi2 with fp32 accumulation. The default conv compute config moves the
  trajectory by about 0.7 m.
- Per-op device time is about 4.4 ms over ~600 ops. That is ~10% of a rough roofline
  (~0.45 ms, DRAM-bound). The rest is small-op overhead and conv efficiency at batch 1.
