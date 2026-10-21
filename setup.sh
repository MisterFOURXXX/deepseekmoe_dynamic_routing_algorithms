#!/bin/bash

# Automatically resolve the directory containing setup.sh and setup.py
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Set repository root variable
REPO_ROOT="./deepseekmoe_dynamic_routing_algorithms"   # Example : "/content/deepseekmoe_dynamic_routing_algorithms"


#Navigating to Repository Root
cd "$REPO_ROOT" 
# Install system dependencies
sudo apt-get update -qq
# Clean existing environment packages (or uninstall old torch/transformers)
pip uninstall -y transformers torch tokenizers torchvision torchaudio huggingface-hub
# Upgrade core build tools
pip install --upgrade pip setuptools wheel
# PyTorch with CUDA
pip install torch==2.13.0 torchvision==0.28.0 # --index-url https://download.pytorch.org/whl/cu130
# Python packages
pip install -r requirements.txt
echo "Environment setup complete!"

# Restart kernel to ensure all changes take effect
exit 0