#!/usr/bin/env bash
# One-time environment setup on the EC2 box (AWS Deep Learning AMI, torch preinstalled in /opt/pytorch).
# Deliberately does NOT create a venv and does NOT reinstall torch: the build is matched to the driver.
#   bash ~/datascan/remote_setup.sh
set -euo pipefail

source /opt/pytorch/bin/activate
cd "$HOME"

test -d EveNet-Lite || git clone -q https://github.com/Dekiofsa/EveNet-Lite.git
pip install -q --no-deps evenet-core==0.3.0            # pins numpy==1.26.4, which we ignore (same as on Colab)
pip install -q --no-deps -e "$HOME/EveNet-Lite"
# runtime dependencies the Colab notebooks installed; torch is already satisfied, so pip leaves it alone
pip install -q lightning torchmetrics scipy scikit-learn wandb huggingface-hub pyyaml rich opt-einsum pyarrow tqdm regex matplotlib

grep -q 'EVENET_LITE_DIR' "$HOME/.bashrc" || echo 'export EVENET_LITE_DIR="$HOME/EveNet-Lite"' >> "$HOME/.bashrc"
export EVENET_LITE_DIR="$HOME/EveNet-Lite"
mkdir -p "$HOME/datascan/logs"

python - <<'EOF'
import os, sys
sys.path.insert(0, os.path.expanduser("~/EveNet-Lite"))
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
import evenet_lite, lightning, torchmetrics
from nsbi.train import build_classifier, build_classifier_ssl, SEQ_DIM, GLOBAL_DIM
from nsbi.data import make_classifier_data, FEATURE_NAMES
sys.path.insert(0, os.path.expanduser("~/datascan"))
import nsbi_scan as ns
print("EveNet-Lite", evenet_lite.__file__)
print("nsbi_scan   ", ns.__file__)
print("setup OK")
EOF
