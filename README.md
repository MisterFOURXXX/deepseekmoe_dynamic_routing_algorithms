# Enhancing DeepSeek Architecture-Based Chatbots Using Dynamic Routing Algorithms

This repository contains the official implementation of the research paper "Enhancing DeepSeek Architecture-Based Chatbots Using Dynamic Routing Algorithms". We integrate DYNMoE's dynamic Top-Any gating with the DeepSeekMoE architecture to overcome the computational inefficiencies of fixed Top-K routing, enabling adaptive expert activation based on token complexity.

The Mixture-of-Experts (MoE) architecture has become fundamental for scaling large language models (LLMs), with DeepSeekMoE achieving expert specialization through fine-grained segmentation and shared expert isolation. However, the reliance on fixed Top-K routing causes persistent load imbalance and computational inefficiency—simple tokens waste resources through over-activation while complex tokens lack sufficient capacity.

We address these limitations by integrating DYNMoE's dynamic routing algorithm with the DeepSeekMoE architecture. Our approach replaces fixed Top-K routing with adaptive Top-Any gating, where each token autonomously determines its number of activated routed experts. We incorporate DYNMoE's auxiliary loss to promote stable expert load balance and prevent routing collapse, along with adaptive expert tuning that dynamically adjusts the expert pool during training. Our comprehensive evaluation demonstrates that the integrated architecture maintains or improves baseline performance while significantly reducing computational resource usage during both training and inference.

---

### Key Contributions

- **Dynamic Routing Integration** – Replaced DeepSeekMoE's fixed Top‑K routing with DYNMoE's Top‑Any gating, enabling token‑level adaptive expert activation.
- **Enhanced Auxiliary Loss** – Incorporated diversity and simplicity losses to promote expert orthogonality and stable routing.
- **Adaptive Expert Tuning** – Implemented automatic expert pool resizing during training to match data complexity.
- **Comprehensive Evaluation** – Validated improvements in computational efficiency while maintaining performance on MultiWOZ 2.3.

---

### Repository Structure

```text
deepseekmoe_dynamic_routing_algorithms/
├── checkpoints/                          # Saved models (created during runs)
├── dataset/                              # MultiWOZ dataset (place during setup)
│   └── Info.txt
├── notebooks/                            # Jupyter notebooks for experiments
│   ├── 01_training_comparison.ipynb      # Training comparison all model architectures
│   ├── 02_fine_tuning_comparison.ipynb   # Fine‑tuning from pre‑trained checkpoints comparison all three architectures
│   └── 03_evaluation_comparison.ipynb    # Evaluation and comparison all model architectures
├── source/
│   ├── data_preprocessing/               # Dataset loading and preprocessing
│   │   ├── config.py
│   │   └── preprocessing.py
│   ├── deepseek_baseline/                # Original DeepSeekMoE (fixed Top‑K)
│   │   ├── config.py
│   │   └── model.py
│   ├── DYNMoE_baseline/                  # Pure DYNMoE implementation (Phi‑2 based with DYNMoE)
│   │   ├── adaptive_tuning.py
│   │   ├── config.py
│   │   └── model.py
│   ├── deepseek_dynamics_routing/        # Our integrated prototype
│   │   ├── adaptive_tuning.py
│   │   ├── config.py
│   │   └── model.py
│   ├── evaluation/                       # Evaluation scripts
│   │   ├── config.py
│   │   ├── evaluation_runner.py
│   │   ├── evaluation.py                 # Core evaluation logic
│   │   └── model_loading.py              # Auto‑detect and load model from checkpoint
│   ├── fine_tuning_utils/                # Fine‑tuning utilities
│   │   ├── config.py
│   │   ├── fine_tuning_runner.py
│   │   ├── model_loading.py              # Auto‑detect and load model from checkpoint
│   │   ├── monitoring.py
│   │   ├── save_model.py
│   │   └── summarization.py
│   ├── training_utils/                   # Training utilities
│   │   ├── config.py
│   │   ├── monitoring.py
│   │   ├── save_model.py
│   │   ├── summarization.py
│   │   └── training_runner.py
│   └── memory_utils.py                   # GPU/CPU memory cleanup functions
├── LICENSE
├── README.md
├── requirements.txt                      # Python dependencies
├── setup.py                              # Package installation
└── setup.sh                              # System dependencies + environment setup
```
---

### Getting Started

**Step 1: Clone the Repository**

```bash
git clone https://github.com/yourusername/deepseekmoe_dynamic_routing_algorithms.git
```

**Step 2: Setup System Environment and Install Python Packages**

```bash
cd ./deepseekmoe_dynamic_routing_algorithms
bash setup.sh
```

**Step 3: Install the Packages and Modules in Repository**

```bash
cd ./deepseekmoe_dynamic_routing_algorithms
pip install -e .
```

Note : If there is errors in import modules in the repositories during experiment, please use the command below.

```bash
pip install -e ./deepseekmoe_dynamic_routing_algorithms
```

**Step 4: Restart Python Kernel**

After installation, restart your Python kernel to ensure all packages are properly loaded.

```bash
exit 0
```

**Step 5: Execute Data Preprocessing**

**Step 5.1: Configure Paths**

Update the paths for data preprocessing and to save dataset `./source/data_preprocessing/config.py`:
```python
DEFAULT_ZIP_PATH = "./deepseekmoe_dynamic_routing_algorithms/dataset/MultiWOZ-coref/MultiWOZ2_3.zip"
DEFAULT_TRAIN_PATH = "./deepseekmoe_dynamic_routing_algorithms/dataset/train_sequences.txt"
DEFAULT_FINE_PATH = "./deepseekmoe_dynamic_routing_algorithms/dataset/fine_sequences.txt"
DEFAULT_EVAL_PATH = "./deepseekmoe_dynamic_routing_algorithms/dataset/eval_sequences.txt"
```

**Step 5.2: Run Preprocessing**

```bash
cd ./deepseekmoe_dynamic_routing_algorithms/source/data_preprocessing
python3 preprocessing.py
```

**Step 6: Configure Model Training, Fine Tuning, and Evaluation Paths**

Update the output paths in `source/training_utils/config.py`:

```python
OUTPUT_BASELINE = "./deepseekmoe_dynamic_routing_algorithms/checkpoints/baseline"
OUTPUT_DYNMOE_BASE = "./deepseekmoe_dynamic_routing_algorithms/checkpoints/dynmoe_baseline"
OUTPUT_DEEPSEEK_DYNMOE = "./deepseekmoe_dynamic_routing_algorithms/checkpoints/dynmoe_routing"
```

The repository will automatically config all training, fine-tuning, and evaluation paths for all models.

---

### Running Experiments

All experiments are organized as Jupyter notebooks in the `./deepseekmoe_dynamic_routing_algorithms/notebooks/` directory, which are training comparison, fine tuning comparison, and evaluation comparison.

**Notebook Details**

**01_training_comparison.ipynb – Pre‑training**

Trains all three models **from scratch** on the MultiWOZ 2.3 dataset:

- **DeepSeekMoE Baseline**: DeepSeekMoE with fixed Top‑K (K=2)
- **DYNMoE Baseline**: DYNMoE (Phi‑2 architecture) with Top‑Any gating, following "Dynamic Mixture of Experts: An Auto-Tuning Approach for Efficient Transformer Models" paper
- **Prototype**: Our integrated DeepSeekMoE with DYNMoE routing

**Outputs:** Checkpoints saved in `checkpoints/baseline/`, `checkpoints/dynmoe_baseline/`, and `checkpoints/dynmoe_routing/`.

**02_fine_tuning_comparison.ipynb – Fine‑tuning**

Loads the pre‑trained checkpoints and fine‑tunes them on the same dataset with a lower learning rate and smaller batch size.

**Outputs:** Fine‑tuned models saved in `checkpoints/baseline-ft/`, `checkpoints/dynmoe_baseline-ft/`, and `checkpoints/dynmoe_routing-ft/`.

**03_evaluation_comparison.ipynb – Evaluation**

Evaluates all three fine‑tuned models on the test set, computing:
- Perplexity
- ROUGE-1/2/L and BLEU scores
- Expert load balance (MaxVio global and batch)
- Computational metrics (FLOPs, active parameters, average activated experts)
- Hardware usage (GPU memory, CPU usage)

**Customising Hyperparameters**

All configurable parameters are centralised in the `config.py` files under each model architecture and utility folder (e.g., `source/training_utils/config.py` and `source/DYNMoE_baseline/config.py`).


**Memory Management**

The repository provides a dedicated `source/memory_utils.py` module to clean up GPU and CPU memory between experiments. The `cleanup_trainer(trainer)` function deletes the model and trainer objects, clears CUDA caches, and destroys the distributed process group. The `clear_cached_data()` function releases the global dataset cache. All training and fine‑tuning runners automatically call these cleanup functions after each run, making it safe to run multiple experiments sequentially without restarting the kernel.

However, to ensure fair results of each model comparison and avoid errors, restart kernel after each model is required. 

**Important Note** 

You must restart the kernel before running each model (DeepSeekMoE Baseline, DYNMoE Baseline, and Prototype) in each notebook to:
- Clear GPU memory
- Remove cached tensors
- Avoid out-of-memory (OOM) errors
- Prevent model/tokenizer overlap issues

---

### Citation

If you use this code or our findings in your research, please cite our paper:

```bibtex
@inproceedings{2025enhancing,
  title={Enhancing DeepSeek Architecture-Based Chatbots Using Dynamic Routing Algorithms},
  author={Mohammad Mahdavi and Apiwit Karnjanavivin},
  booktitle={Proceedings of the Association for Computing Machinery (ACM)},
  year={2025},
  note={Accepted for publication}
}
```

---

### License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

---

### Contact

- **Mohammad Mahdavi** – [mohammad.mahdavi@gisma.com](mailto:mohammad.mahdavi@gisma.com)
- **Apiwit Karnjanavivin** – [Apiwit.Karnjanavivin@gisma-student.com](mailto:Apiwit.Karnjanavivin@gisma-student.com)

---

### Disclaimer

This repository contains experimental software developed for research purposes. The code is provided "as is" without warranty of any kind, either expressed or implied.
