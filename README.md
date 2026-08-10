# Enhancing DeepSeek Architecture-Based Chatbots Using Dynamic Routing Algorithms

This repository contains the official implementation of "Enhancing DeepSeek Architecture-Based Chatbots Using Dynamic Routing Algorithms". We integrate DYNMoE's dynamic Top-Any gating with the DeepSeekMoE architecture to overcome the computational inefficiencies of fixed Top-K routing, enabling adaptive expert activation based on token complexity.

The Mixture-of-Experts (MoE) architecture has become fundamental for scaling large language models (LLMs), with DeepSeekMoE achieving expert specialization through fine-grained segmentation and shared expert isolation. However, the reliance on fixed Top-K routing causes persistent load imbalance and computational inefficiency—simple tokens waste resources through over-activation while complex tokens lack sufficient capacity.

We address these limitations by integrating DYNMoE's dynamic routing algorithm with the DeepSeekMoE architecture. Our approach replaces fixed Top-K routing with adaptive Top-Any gating, where each token autonomously determines its number of activated routed experts. We incorporate DYNMoE's auxiliary loss to promote stable expert load balance and prevent routing collapse, along with adaptive expert tuning that dynamically adjusts the expert pool during training. Our comprehensive evaluation demonstrates that the integrated architecture maintains or improves baseline performance while significantly reducing computational resource usage during both training and inference.

**Repository Structure**

```text
deepseekmoe_dynamic_routing_algorithms/
├── dataset/
│   └── Info.txt                           # Dataset information
├── notebooks/
│   ├── 01_training_comparison.ipynb      # Pre-training comparison
│   ├── 02_fine_tuning_comparison.ipynb   # Fine-tuning comparison
│   └── 03_evaluation_comparison.ipynb    # Evaluation comparison
├── source/
│   ├── __init__.py
│   ├── deepseek_baseline/                # Baseline DeepSeekMoE
│   │   ├── config.py
│   │   └── model.py
│   ├── DYNMoE_baseline/                  # DYNMoE baseline
│   │   ├── adaptive_tuning.py
│   │   ├── config.py
│   │   └── model.py
│   ├── deepseek_dynamics_routing/        # Our integrated prototype
│   │   ├── adaptive_tuning.py
│   │   ├── config.py
│   │   └── model.py
│   ├── data_preprocessing.py             # MultiWOZ preprocessing
│   ├── training_utils/                   # Training utilities
│   │   ├── config.py
│   │   ├── monitoring.py
│   │   ├── save_model.py
│   │   └── summarization.py
│   ├── fine_tuning_utils/                # Fine-tuning utilities
│   │   ├── config.py
│   │   ├── model_loading.py
│   │   ├── monitoring.py
│   │   ├── save_model.py
│   │   └── summarization.py
│   └── evaluation/
│       ├── config.py
│       ├── evaluation.py
│       └── model_loading.py
├── checkpoints/
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
└── setup.sh                              # System dependencies + Python packages
```

**Quick Start**

**Step 1: Clone the Repository**

```bash
git clone https://github.com/yourusername/deepseekmoe_dynamic_routing_algorithms.git
```

**Step 2: Setup system environment and Install Python packages**

```bash
cd /kaggle/working/deepseekmoe_dynamic_routing_algorithms
bash setup.sh
```

**Step 3: Restart Python Kernel**

```bash
exit 0
```

cd /kaggle/working/deepseekmoe_dynamic_routing_algorithms

# pip install -e .
!pip install -e /kaggle/working/deepseekmoe_dynamic_routing_algorithms

**Step 4: Execute Data Preprocessing**

**Step 4.1:**

Revise path for data preprocessing and save to dataset

DEFAULT_ZIP_PATH = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/dataset/MultiWOZ-coref/MultiWOZ2_3.zip"
DEFAULT_TRAIN_PATH = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/dataset/train_sequences.txt"
DEFAULT_FINE_PATH = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/dataset/fine_sequences.txt"
DEFAULT_EVAL_PATH = "/kaggle/working/deepseekmoe_dynamic_routing_algorithms/dataset/eval_sequences.txt"

**Step 4.2:**

```bash
cd /kaggle/working/deepseekmoe_dynamic_routing_algorithms/source/data_preprocessing
python3 preprocessing.py
```

Changed data path, model training, fine tuning, and evaluation at configs of each folder source

