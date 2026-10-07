#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Build the venv for navsim_pdm_eval.py: NAVSIM devkit + nuplan-devkit + pinned deps.
#
#   meanfuser/scripts/setup_navsim_venv.sh <navsim-root>
#
#   <navsim-root>/MeanFuser       upstream devkit, cloned at a pinned commit if absent
#   <navsim-root>/nuplan-devkit   cloned at v1.2 if absent
#   <navsim-root>/venv            the venv
#
# ttnn and torch are not installed here. NAVSIM and tt-metal pin conflicting versions
# (protobuf, scikit-learn, opencv), so the venv loads them from $TT_METAL_HOME/python_env at
# interpreter start, after its own packages. One venv serves any tt-metal checkout.
set -euo pipefail

root=$(realpath -m "${1:?usage: $0 <navsim-root>}")
here=$(cd "$(dirname "$0")" && pwd)

clone() {
    local url=$1 dir=$2 commit=$3
    if [ ! -d "$dir/.git" ]; then
        git clone --quiet "$url" "$dir"
        git -C "$dir" checkout --quiet "$commit"
    fi
}

mkdir -p "$root"
clone https://github.com/wjl2244/MeanFuser.git "$root/MeanFuser" 8de8ba6244834645192e318dcc437d124cfd6872
clone https://github.com/motional/nuplan-devkit.git "$root/nuplan-devkit" ce3c323af01c0d7ec5672f7832ef53f9c679aab0

py=$root/venv/bin/python
[ -x "$py" ] || uv venv --quiet --python 3.10 "$root/venv"
# --no-deps: the pin list is the full closure. Resolving would pull torch in through
# pytorch-lightning/torchmetrics and shadow the tt-metal torch.
uv pip install --quiet --python "$py" --no-deps -r "$here/navsim_requirements.txt"
uv pip install --quiet --python "$py" --no-deps -e "$root/nuplan-devkit" -e "$root/MeanFuser"

site=$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
cat > "$site/ttmetal_env.py" <<'EOF'
import os
import site
import sys

home = os.environ.get("TT_METAL_HOME")
if home:
    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site.addsitedir(os.path.join(home, "python_env", "lib", py, "site-packages"))
EOF
echo "import ttmetal_env" > "$site/zz_ttmetal_env.pth"

echo "NAVSIM venv ready: $root/venv"
echo "  export TT_METAL_HOME=<tt-metal checkout>; source $root/venv/bin/activate"
