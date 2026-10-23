#!/bin/bash
set -e

# Resolve directory containing setup.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------
# >>>>>>>>>> EDIT ONLY THESE TWO LINES <<<<<<<<<<<<<<<<<<<<<<<
# ------------------------------------------------------------
REPO_ROOT="./deepseekmoe_dynamic_routing_algorithms"   # e.g. "/content/deepseekmoe_dynamic_routing_algorithms"
HF_TOKEN="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"
# ------------------------------------------------------------

cd "$REPO_ROOT"
REPO_ROOT_ABS="$(pwd)"

# Derive paths from REPO_ROOT
DATASET_ROOT="$REPO_ROOT_ABS/dataset"
OUTPUT_BASELINE="$REPO_ROOT_ABS/checkpoints/baseline"
OUTPUT_DEEPSEEK_DYNMOE="$REPO_ROOT_ABS/checkpoints/dynmoe_routing"

echo "Resolved REPO_ROOT           = $REPO_ROOT_ABS"
echo "DATASET_ROOT                 = $DATASET_ROOT"
echo "OUTPUT_BASELINE              = $OUTPUT_BASELINE"
echo "OUTPUT_DEEPSEEK_DYNMOE       = $OUTPUT_DEEPSEEK_DYNMOE"

# Overwrite the specific assignments in the two Python config files.
export REPO_ROOT_ABS DATASET_ROOT OUTPUT_BASELINE OUTPUT_DEEPSEEK_DYNMOE HF_TOKEN
python3 - <<'PY'
import os
import re

CFG_DATA  = os.path.join(os.environ["REPO_ROOT_ABS"], "source/data_preprocessing/config.py")
CFG_TRAIN = os.path.join(os.environ["REPO_ROOT_ABS"], "source/training_utils/config.py")


def py_quote(v: str) -> str:
    """Wrap value as a Python double-quoted string, escaping \\ and \"."""
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def replace_assignment(path: str, name: str, value: str) -> None:
    """Replace only the `NAME = ...` definition line in `path`."""
    if not os.path.isfile(path):
        print(f"  WARN: {path} not found; skipping {name}")
        return
    with open(path, "r") as f:
        src = f.read()
    pattern = re.compile(rf'^({re.escape(name)}\s*=\s*).*$', re.MULTILINE)
    new_src, n = pattern.subn(lambda m: m.group(1) + py_quote(value), src)
    if n == 0:
        print(f"  WARN: {name} not found in {path}")
        return
    with open(path, "w") as f:
        f.write(new_src)
    print(f"  set {name:22s} = {value}")


print("\nUpdating config files ...")
replace_assignment(CFG_DATA,  "HF_TOKEN",               os.environ["HF_TOKEN"])
replace_assignment(CFG_DATA,  "DATASET_ROOT",           os.environ["DATASET_ROOT"])
replace_assignment(CFG_DATA,  "REPO_ROOT",              os.environ["REPO_ROOT_ABS"])
replace_assignment(CFG_TRAIN, "OUTPUT_BASELINE",        os.environ["OUTPUT_BASELINE"])
replace_assignment(CFG_TRAIN, "OUTPUT_DEEPSEEK_DYNMOE", os.environ["OUTPUT_DEEPSEEK_DYNMOE"])
PY

# Install packages 
sudo apt-get update -qq
pip uninstall -y transformers torch tokenizers torchvision torchaudio huggingface-hub
pip install --upgrade pip setuptools wheel
pip install torch==2.13.0 torchvision==0.28.0       # PyTorch Version 2.11.0 also compatible : pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
pip install -r requirements.txt
echo "Environment setup complete!"
exit 0