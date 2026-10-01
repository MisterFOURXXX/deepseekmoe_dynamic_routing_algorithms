#!/bin/bash

# Automatically resolve the directory containing setup.sh and setup.py
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Set repository root variable
REPO_ROOT="/content/deepseekmoe_dynamic_routing_algorithms"

#Navigating to Repository Root
cd "$REPO_ROOT" 
# Install system dependencies
sudo apt-get update -qq
sudo apt-get install -y libaio-dev curl python3-dev -qq
# Clean existing environment packages (or uninstall old torch/transformers)
python3 -m pip uninstall -y transformers torch tokenizers torchvision torchaudio huggingface-hub || true
# Upgrade core build tools
python3 -m pip install --upgrade pip setuptools wheel
# PyTorch with CUDA
python3 -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0  #--index-url https://download.pytorch.org/whl/cu128
# Python packages
python3 -m pip install -r requirements.txt
echo "Environment setup complete!"

# Move into the dataset folder of your project
cd "$REPO_ROOT/dataset"

# Clone the MultiWOZ coreference repository
git clone https://github.com/lexmen318/MultiWOZ-coref.git

echo "Download dataset complete!"

# Restart kernel to ensure all changes take effect
exit 0