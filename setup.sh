#!/bin/bash
# System dependencies
sed -i 's/archive.ubuntu.com/mirrors.kernel.org/g' /etc/apt/sources.list
apt-get update -qq && apt-get install -y libaio-dev -qq

# PyTorch with CUDA
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# Python packages
pip install -r requirements.txt

echo "Environment setup complete!"

# Move into the dataset folder of your project
cd /kaggle/working/deepseekmoe_dynamic_routing_algorithms/dataset

# Clone the MultiWOZ coreference repository
git clone https://github.com/lexmen318/MultiWOZ-coref.git

echo "Download dataset complete!"