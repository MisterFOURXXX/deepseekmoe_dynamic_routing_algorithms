import os
import sys
repo_path =  ".."
os.chdir(repo_path)                 # Move into the repo
sys.path.insert(0, os.getcwd())     # Ensure the repo root is on sys.path

# Default paths (adjust as needed)
DEFAULT_ZIP_PATH = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/dataset/MultiWOZ-coref/MultiWOZ2_3.zip"
DEFAULT_TRAIN_PATH = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/dataset/train_sequences.txt"
DEFAULT_FINE_PATH = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/dataset/fine_sequences.txt"
DEFAULT_EVAL_PATH = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/dataset/eval_sequences.txt"