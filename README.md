# Enhancing DeepSeek Architecture-Based Chatbots Using Dynamic Routing Algorithms

This repository contains the official implementation of "Enhancing DeepSeek Architecture-Based Chatbots Using Dynamic Routing Algorithms". We integrate DYNMoE's dynamic Top-Any gating with the DeepSeekMoE architecture to overcome the computational inefficiencies of fixed Top-K routing, enabling adaptive expert activation based on token complexity.

The Mixture-of-Experts (MoE) architecture has become fundamental for scaling large language models (LLMs), with DeepSeekMoE achieving expert specialization through fine-grained segmentation and shared expert isolation. However, the reliance on fixed Top-K routing causes persistent load imbalance and computational inefficiency—simple tokens waste resources through over-activation while complex tokens lack sufficient capacity.

We address these limitations by integrating DYNMoE's dynamic routing algorithm with the DeepSeekMoE architecture. Our approach replaces fixed Top-K routing with adaptive Top-Any gating, where each token autonomously determines its number of activated routed experts. We incorporate DYNMoE's auxiliary loss to promote stable expert load balance and prevent routing collapse, along with adaptive expert tuning that dynamically adjusts the expert pool during training. Our comprehensive evaluation demonstrates that the integrated architecture maintains or improves baseline performance while significantly reducing computational resource usage during both training and inference.

**Repository Structure**

```text
deepseekmoe_dynamic_routing_algorithms/
├── datasets/
├── notebooks/
│   ├── 01_training_comparison.ipynb
│   ├── 02_fine_tuning_comparison.ipynb
│   └── 03_evaluation_comparison.ipynb
├── source/
│   ├── __init__.py 
│   ├── deepseek_baseline/
│   │   ├── config.py                        # DeepseekConfig (baseline)
│   │   └── model.py                         # Full model architecture (baseline)
│   ├── deepseek_dynamics_routing/
│   │   ├── config.py                        # DeepseekConfig (with max_routed_experts)
│   │   ├── model.py                         # Full model architecture (DYNMoE)
│   │   └── adaptive_tuning.py               # AdaptiveExpertTuningCallback
│   ├── data_preprocessing.py
│   ├── training_utils/
│   │   ├── configurations.py
│   │   ├── monitoring.py
│   │   ├── save_model.py                    # Save model + tokenizer
│   │   └── summarization.py                 # Training summary table
│   ├── fine_tuning_utils/
│   │   ├── model_loading.py
│   │   ├── configurations.py
│   │   ├── monitoring.py
│   │   ├── save_model.py                    # Save model + tokenizer
│   │   └── summarization.py                 # Training summary table
│   └── evaluation.py
│       ├── model_loading.py
│       └── evaluation.py
├── checkpoints/
├── README.md
├── requirements.txt
├── setup.py
└── setup.sh                  # System dependencies + PyTorch + pip install
```

cd /kaggle/working/deepseekmoe_dynamic_routing_algorithms

**Step 1: Setup system environment and Install Python packages**
cd /kaggle/working/deepseekmoe_dynamic_routing_algorithms
!bash setup.sh

**Step 2: Restart Python Kernel**
!kill -9 $(pgrep -f ipykernel_launcher)

**Step 3: Execute Training Notebook**
!jupyter nbconvert --to notebook --execute /kaggle/working/deepseekmoe_dynamic_routing_algorithms/notebook/01_training_comparison.ipynb

evaluation/fine_tuning_utils/training_utils