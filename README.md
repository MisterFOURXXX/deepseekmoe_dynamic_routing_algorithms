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
!bash setup.sh

**Step 2: Restart Python Kernel**
!kill -9 $(pgrep -f ipykernel_launcher)

**Step 3: Execute Training Notebook**
!jupyter nbconvert --to notebook --execute /kaggle/working/deepseekmoe_dynamic_routing_algorithms/notebook/01_training_comparison.ipynb

evaluation/fine_tuning_utils/training_utils