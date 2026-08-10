import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)   # <-- insert the project root

# Default paths – these are absolute and unchanged
DEFAULT_ZIP_PATH = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/dataset/MultiWOZ-coref/MultiWOZ2_3.zip"
DEFAULT_TRAIN_PATH = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/dataset/train_sequences.txt"
DEFAULT_FINE_PATH = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/dataset/fine_sequences.txt"
DEFAULT_EVAL_PATH = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/dataset/eval_sequences.txt"