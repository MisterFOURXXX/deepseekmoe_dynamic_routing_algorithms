#!/bin/bash

# Automatically resolve the directory containing setup.sh and setup.py
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Set repository root variable
REPO_ROOT="/kaggle/working/deepseekmoe_dynamic_routing_algorithms"

#Navigating to Repository Root
cd "$REPO_ROOT"

# System dependencies
sed -i 's/archive.ubuntu.com/mirrors.kernel.org/g' /etc/apt/sources.list
apt-get update -qq && apt-get install -y libaio-dev -qq

# PyTorch with CUDA
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# Python packages
pip install -r requirements.txt

#pip install --upgrade pip setuptools wheel

cd "$REPO_ROOT"

# Registering Package (pip install -e .)/ Linking Repository Paths (pip install -e .)
pip install -e ..
#pip install -e "$REPO_ROOT"           # If there are any errors, try this command instead of the previous one.

echo "Environment setup complete!"

# Move into the dataset folder of your project
cd "$REPO_ROOT/dataset"

# Clone the MultiWOZ coreference repository
git clone https://github.com/lexmen318/MultiWOZ-coref.git

echo "Download dataset complete!"

exit 0