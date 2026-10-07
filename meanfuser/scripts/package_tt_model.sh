#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Stage MeanFuser as a tt-model v6 thin bundle (kind tt-dit-server) from this checkout.
#
#   meanfuser/scripts/package_tt_model.sh <out-dir> [org/name to push] [more package-thin args]
#
# Needs a built third_party/tt-metal (build_metal.sh), uv, and the tt-model CLI. The bundle
# carries a ttnn wheel of the pinned tt-metal (no index has it) and a meanfuser wheel. It
# carries no weights: the server reads MF_CHECKPOINT_PATH and MF_GMN_MEAN_PATH at startup.
set -euo pipefail

out=$(realpath -m "${1:?usage: $0 <out-dir> [package-thin args]}")
shift
repo=$(cd "$(dirname "$0")/../.." && pwd)
wheels=$(mktemp -d)
trap 'rm -rf "$wheels" "$repo/build" "$repo/meanfuser.egg-info"' EXIT

# ttnn from the existing build tree; setup.py packages it without rebuilding.
TT_FROM_PRECOMPILED_DIR="$repo/third_party/tt-metal" \
    uv build --quiet --wheel --python 3.10 --out-dir "$wheels" "$repo/third_party/tt-metal"
uv build --quiet --wheel --out-dir "$wheels" "$repo"

ttnn_whl=$(ls "$wheels"/ttnn-*.whl)
meanfuser_whl=$(ls "$wheels"/meanfuser-*.whl)
ttnn_ver=$(basename "$ttnn_whl" | cut -d- -f2)
meanfuser_ver=$(basename "$meanfuser_whl" | cut -d- -f2)

# Versions validated together; torch must be the CPU build (tt-model adds its index).
cat > "$wheels/requirements.txt" <<EOF
ttnn==$ttnn_ver
meanfuser==$meanfuser_ver
torch==2.11.0+cpu
torchvision==0.26.0+cpu
timm==1.0.30
diffusers==0.38.0
numpy==1.26.4
pillow==12.3.0
fastapi==0.142.2
pydantic==2.9.2
uvicorn==0.54.0
EOF

tt-model package-thin \
    --model-py "$repo/meanfuser/server/app.py" \
    --kind tt-dit-server --app meanfuser.server.app:app \
    --requirements "$wheels/requirements.txt" \
    --models-wheel "$ttnn_whl" --models-wheel "$meanfuser_whl" \
    --arch blackhole --mesh P150 --device-count 1 --python 3.10 \
    --name meanfuser --out "$out" "$@"
