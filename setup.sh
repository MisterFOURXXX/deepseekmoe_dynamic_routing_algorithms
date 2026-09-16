#!/bin/bash

# Automatically resolve the directory containing setup.sh and setup.py
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Set repository root variable
#REPO_ROOT="/home/ubuntu/deepseekmoe_dynamic_routing_algorithms"
REPO_ROOT="/kaggle/working/deepseekmoe_dynamic_routing_algorithms"

#Navigating to Repository Root
cd "$REPO_ROOT"

# System dependencies
#sed -i 's/archive.ubuntu.com/mirrors.kernel.org/g' /etc/apt/sources.list
sudo apt-get update -qq && sudo apt-get install -y libaio-dev -qq
# sudo add-apt-repository ppa:deadsnakes/ppa -y
# sudo apt install -y python3.11 python3.11-venv python3.11-dev
# sudo apt update

python -m pip install --upgrade pip setuptools wheel

# PyTorch with CUDA
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# Python packages
python -m pip install -r requirements.txt

echo "Environment setup complete!"

# Move into the dataset folder of your project
cd "$REPO_ROOT/dataset"

# Clone the MultiWOZ coreference repository
git clone https://github.com/lexmen318/MultiWOZ-coref.git

echo "Download dataset complete!"

# Restart kernel to ensure all changes take effect
exit 0