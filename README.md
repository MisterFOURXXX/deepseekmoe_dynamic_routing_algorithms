# Enhancing DeepSeek Architecture-Based Chatbots Using Dynamic Routing Algorithms

This repository contains the official implementation of the research paper "Enhancing DeepSeek Architecture-Based Chatbots Using Dynamic Routing Algorithms". We integrate DYNMoE's dynamic Top-Any gating with the DeepSeekMoE architecture to overcome the computational inefficiencies of fixed Top-K routing, enabling adaptive expert activation based on token complexity.

The Mixture-of-Experts (MoE) architecture has become fundamental for scaling large language models (LLMs), with DeepSeekMoE achieving expert specialization through fine-grained segmentation and shared expert isolation. However, the reliance on fixed Top-K routing causes persistent load imbalance and computational inefficiency—simple tokens waste resources through over-activation while complex tokens lack sufficient capacity.

We address these limitations by integrating DYNMoE's dynamic routing algorithm with the DeepSeekMoE architecture. Our approach replaces fixed Top-K routing with adaptive Top-Any gating, where each token autonomously determines its number of activated routed experts. We incorporate DYNMoE's auxiliary loss to promote stable expert load balance and prevent routing collapse, along with adaptive expert tuning that dynamically adjusts the expert pool during training. Our comprehensive evaluation demonstrates that the integrated architecture maintains or improves baseline performance while significantly reducing computational resource usage during both training and inference.

---

### Key Contributions

- **Dynamic Routing Integration** – Replaced DeepSeekMoE's fixed Top‑K routing with DYNMoE's Top‑Any gating, enabling token‑level adaptive expert activation.
- **Enhanced Auxiliary Loss** – Incorporated diversity and simplicity losses to promote expert orthogonality and stable routing.
- **Adaptive Expert Tuning** – Implemented automatic expert pool resizing during training to match data complexity.
- **Comprehensive Evaluation** – Evaluated on computational efficiency and model performance during training and fine tuning using the SlimPajama dataset, combined with benchmark evaluation for LLMs inference on MiniPile benchmark.

---

### Repository Structure

```text
deepseekmoe_dynamic_routing_algorithms/
├── checkpoints/                          # Saved models (created during runs)
├── dataset/                              # SlimPajama & benchmark datasets (place during setup)
│   └── Info.txt
├── notebooks/                            # Jupyter notebooks for experiments
│   ├── 01_training_comparison.ipynb      # Training comparison of model architectures
│   ├── 02_fine_tuning_comparison.ipynb   # Fine‑tuning from pre‑trained checkpoints comparison
│   └── 03_evaluation_comparison.ipynb    # Evaluation and comparison of model architectures
├── source/
│   ├── data_preprocessing/               # Dataset loading and preprocessing
│   │   ├── config.py
│   │   └── preprocessing.py
│   ├── deepseek_baseline/                # Original DeepSeekMoE (fixed Top‑K)
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

**Step 2: Change Download and Packages Configure Paths**

In `setup.sh` at `REPO_ROOT` variable, it has to change the path to your project location, and update the paths and your HuggingFace Token for downloading the SlimPajama and benchmark datasets

```bash
REPO_ROOT="./deepseekmoe_dynamic_routing_algorithms"   # Ex. cd /content/working/deepseekmoe_dynamic_routing_algorithms
HF_TOKEN="hf_..."                                       # Replace with your HUggingFace access token (READ token)
```

**Step 3: Setup System Environment and Install Python Packages**

**Step 3.1:** Create virtual environments

```bash
cd ./deepseekmoe_dynamic_routing_algorithms            # Ex. cd /content/working/deepseekmoe_dynamic_routing_algorithms
python3 -m venv .venv
source .venv/bin/activate
python --version                                       # should print Python 3.xx.xx
```

**Step 3.2:** Setup all environments and packages

```bash
cd ./deepseekmoe_dynamic_routing_algorithms            # Ex. cd /content/working/deepseekmoe_dynamic_routing_algorithms
chmod +x setup.sh
bash setup.sh
```

If there is any package import errors or environments errors when running notebook experiments, please try:

```bash
cd ./deepseekmoe_dynamic_routing_algorithms            # Ex. cd /content/working/deepseekmoe_dynamic_routing_algorithms
python -m pip install --upgrade pip setuptools wheel
# Uninstall previous version to prevent import errors and prevent conflicts
python -m pip uninstall -y transformers huggingface-hub tokenizers torch torchvision torchaudio 
# PyTorch with CUDA
python -m pip install torch==2.13.0 torchvision==0.28.0       # PyTorch Version 2.11.0 also compatible : pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
# Python packages
python -m pip install -r requirements.txt
```

**Note: Restart kernel after finish running setup.sh**

---

### Running Experiments

All experiments are organized as Jupyter notebooks in the `./deepseekmoe_dynamic_routing_algorithms/notebooks/` directory, which are training comparison, fine tuning comparison, and evaluation comparison.

**Notebook Details**

**01_training_comparison.ipynb – Pre‑training**

Trains both models **from scratch** on the SlimPajama dataset. Both models are validated at the end of each epoch.

- **DeepSeekMoE Baseline**: DeepSeekMoE with fixed Top‑K (K=2).
- **Prototype**: Our integrated DeepSeekMoE with DYNMoE routing. It matches standard transformer hyperparameters with 2 shared experts and an initial pool of 8 routed experts per layer. Routing thresholds are initialized to **-5.95**, dynamically activating between 1 and 6 routed experts with periodic adaptive tuning and an inference fallback safeguard. Our initial configurations of the model architecture were optimized for the SlimPajama dataset.

**Configuration & Data Preprocessing:**

The notebook is organized into three configuration cells:

1. **Training Configurations** – Centralized hyperparameters for training (`MAX_SEQ_LEN`, `LEARNING_RATE`, `PER_DEVICE_BATCH`, etc.) and DeepSpeed (`ds_config`), which is saved to `ds_config.json` so the training runner can load it.
2. **Data Preprocessing** – A dedicated cell calls `run_preprocessing(...)` to build the SlimPajama **training**, **fine-tuning**, and **validation** splits. The dataset is tokenized using the DeepSeekMoE tokenizer (BPE) and truncated to `MAX_SEQ_LEN`. Runtime parameters are passed directly from the notebook:

```python
run_preprocessing(
    max_seq_len=MAX_SEQ_LEN,
    sample_ratio=SAMPLE_RATIO,           # e.g. 0.0003 is equal 0.03 percent of SlimPajama
    validation_ratio=VALIDATION_RATIO,   # e.g. 0.1 is equal 10 percent of HF validation set
    train_split_ratio=TRAIN_SPLIT_RATIO, # e.g. 0.6 is equal 60 percent train and 40 percent fine-tune
    build_slimpajama_data=True,
    build_minipile_data=False,
)
```

3. **Model-Specific Configs** – Passed directly into `run_training(...)`, including `n_shared_experts`, `aux_loss_alpha`, `threshold_init`, `audit_steps`, and other DYNMoE-specific hyperparameters.

**Smart Rebuild:** The preprocessing step tracks the parameters used in `dataset/_preprocessing_meta.json`. If you change `SAMPLE_RATIO` or `MAX_SEQ_LEN` and re-run the cell, the old datasets are automatically deleted and rebuilt. Set `force_rebuild=True` to force a rebuild regardless.

**Outputs:** Checkpoints are saved in `checkpoints/baseline/` and `checkpoints/dynmoe_routing/`.

---

**02_fine_tuning_comparison.ipynb – Fine‑tuning**

Loads the pre‑trained checkpoints from the training notebook and fine‑tunes them on the fine-tuning split. Both models are validated at the end of each epoch.

**Configuration & Outputs:** A centralized configuration cell dictates the fine-tuning hyperparameters (e.g., `MAX_SEQ_LEN=384`, `LEARNING_RATE=6e-5`) and DeepSpeed config (`ds_config.json`). The fine-tuning datasets are already built by the training notebook — no re-preprocessing is required. Fine‑tuned models are saved in `checkpoints/baseline-ft/` and `checkpoints/dynmoe_routing-ft/`.

---

**03_evaluation_comparison.ipynb – Evaluation**

Evaluates both fine‑tuned models and comprehensively measures three key dimensions:

1. **Expert load balance:** Utilizes the Maximal Violation (MaxVio) metric across batches to detect routing collapse and quantify expert load distribution.
2. **Computational efficiency:** Tracks hardware resource usage (GPU/CPU memory and utilization) via PyTorch tooling, NVIDIA-SMI, and psutil. Efficiency is quantified using active versus total parameter ratios, training GFLOPs per step, and inference GFLOP throughput.
3. **Model performance:** Assesses language modeling capability with **Perplexity** on the MiniPile benchmark during inference.

**Configuration & Data Preprocessing:**

The evaluation notebook is split into two configuration cells:

1. **Evaluation Configuration** – Centralized runtime parameters (`MAX_SEQ_LEN`, `BENCH_SAMPLE_RATIO`, `eval_batch_size`, `gen_max_new_tokens`, etc.).
2. **Data Preprocessing** – A dedicated cell builds the MiniPile benchmark subset with the parameters above. The SlimPajama data is skipped since it was already built by the training notebook:

```python
run_preprocessing(
    max_seq_len=MAX_SEQ_LEN,
    bench_sample_ratio=BENCH_SAMPLE_RATIO,  # e.g. 0.05 equal to 5 percent of MiniPile test set
    build_slimpajama_data=False,            # skip — already built by training notebook
    build_minipile_data=True,
)
```

**Model Evaluation:** Both models are then evaluated with the same `run_evaluate(...)` call, passing `MAX_SEQ_LEN`, `eval_batch_size`, and other evaluation parameters directly to the runner.

**Customising Hyperparameters**

While default values are still stored in the `config.py` files under each model architecture and utility folder, the Jupyter Notebooks are now designed for seamless experimentation. You can override training hyperparameters, model configurations, dataset parameters (via `run_preprocessing`), and DeepSpeed settings directly within the notebook cells without editing the source code. The preprocessing system also automatically rebuilds datasets when you change sample ratios or sequence lengths, ensuring training and evaluation always use data matching your current experiment.

**Memory Management**

The repository provides a dedicated `source/memory_utils.py` module to clean up GPU and CPU memory between experiments. The `cleanup_trainer(trainer)` function deletes the model and trainer objects, clears CUDA caches, and destroys the distributed process group. The `clear_cached_data()` function releases the global dataset cache. All training and fine‑tuning runners automatically call these cleanup functions after each run, making it safe to run multiple experiments sequentially without restarting the kernel.

However, to ensure fair results of each model comparison and avoid errors, restarting the kernel after each model is required.

**Important Note**

You must restart the kernel before running each model (DeepSeekMoE Baseline and Prototype) in each notebook, and you **must run the configuration cells** at the top of the notebook before executing any training, fine-tuning, or evaluation command to:

- Load the centralized hyperparameters and `ds_config`.
- Rebuild (or verify) the preprocessed datasets with the current parameters.
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