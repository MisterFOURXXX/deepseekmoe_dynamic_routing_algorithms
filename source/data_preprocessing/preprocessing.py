import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import json
import shutil
import hashlib
from collections import defaultdict, Counter

os.environ["HF_DATASETS_DISABLE_DILL"]          = "1"
os.environ["TOKENIZERS_PARALLELISM"]            = "false"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

from datasets import (
    load_dataset, Dataset, Features, Value, Sequence,
)
from datasets.utils.logging import disable_progress_bar
from transformers import AutoTokenizer
from huggingface_hub import login

from .config import (
    HF_TOKEN, TOKENIZER_NAME, MAX_SEQ_LEN,
    MINORITY_CONFIDENCE, MINORITY_MARGIN_ERROR, BENCH_SAMPLE_RATIO,
    DOMAIN_SHARES, PRETRAIN_SEED, FINETUNE_SEED,
    MINIPILE_TOTAL_ROWS, BOOTSTRAP_ITERATIONS,
    CHI_SQUARE_P_THRESHOLD, KL_DIVERGENCE_THRESHOLD, VOCAB_SIZE,
    DATASET_ROOT, PATHS, DEFAULT_TOKENIZER_PATH, REPO_ROOT,
    META_FILE,
)
from .statistical_sample import get_stratified_sizes

os.environ["HF_DATASETS_CACHE"] = REPO_ROOT + "/hf_cache"
disable_progress_bar()

_TOKENIZER = None

def _ensure_setup():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    login(token=HF_TOKEN)
    os.makedirs(DEFAULT_TOKENIZER_PATH, exist_ok=True)
    _TOKENIZER = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, trust_remote_code=True, use_fast=True,
    )
    _TOKENIZER.pad_token    = _TOKENIZER.eos_token
    _TOKENIZER.pad_token_id = _TOKENIZER.eos_token_id
    _TOKENIZER.save_pretrained(DEFAULT_TOKENIZER_PATH)
    print(f"Tokenizer saved to {DEFAULT_TOKENIZER_PATH}")
    return _TOKENIZER

def _load_meta():
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_meta(meta):
    os.makedirs(os.path.dirname(META_FILE), exist_ok=True)
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

def _needs_rebuild(key, params, force_rebuild):
    if force_rebuild:
        return True
    meta = _load_meta()
    if key not in meta:
        return True
    return meta[key] != params

def _record_built(key, params):
    meta = _load_meta()
    meta[key] = params
    _save_meta(meta)

def _dataset_saved(path):
    if not os.path.isdir(path):
        return False
    if not any(
        os.path.exists(os.path.join(path, m))
        for m in ("dataset_info.json", "state.json")
    ):
        return False
    try:
        from datasets.info import DatasetInfo
        DatasetInfo.from_directory(path)
        return True
    except Exception as e:
        print(f"  WARNING: corrupted saved dataset at {path}: {type(e).__name__}: {e}")
        print(f"  Deleting {path} and re-processing...")
        try:
            shutil.rmtree(path)
        except Exception as rm_e:
            print(f"  Could not delete {path}: {rm_e}")
        return False

def _clear_builder_cache(dataset_name, config_name=None):
    if not dataset_name or "/" not in dataset_name:
        return
    org, _, name = dataset_name.partition("/")
    base = os.path.join(os.environ["HF_DATASETS_CACHE"], f"{org}___{name}")
    if config_name:
        base = os.path.join(base, config_name)
    if os.path.isdir(base):
        shutil.rmtree(base)

def _safe_load_dataset(*args, **kwargs):
    try:
        return load_dataset(*args, **kwargs)
    except TypeError as e:
        msg = str(e).lower()
        if "dataclass" not in msg:
            raise
        dataset_name = args[0] if args else kwargs.get("path")
        config_name  = args[1] if len(args) > 1 else kwargs.get("name")
        _clear_builder_cache(dataset_name, config_name)
        return load_dataset(*args, **kwargs)

def manual_tokenize(text_iter, max_len, batch_size=8192):
    tokenizer = _TOKENIZER
    buf = []
    for text in text_iter:
        buf.append(text)
        if len(buf) >= batch_size:
            out = tokenizer(buf, truncation=True, max_length=max_len,
                            padding=False, return_attention_mask=True)
            for j in range(len(out["input_ids"])):
                yield {"input_ids": out["input_ids"][j],
                       "attention_mask": out["attention_mask"][j]}
            buf = []
    if buf:
        out = tokenizer(buf, truncation=True, max_length=max_len,
                        padding=False, return_attention_mask=True)
        for j in range(len(out["input_ids"])):
            yield {"input_ids": out["input_ids"][j],
                   "attention_mask": out["attention_mask"][j]}

LM_FEATURES = Features({
    "input_ids":      Sequence(Value("int32")),
    "attention_mask": Sequence(Value("int32")),
})

# ------------------------------------------------------------------
# 1D Proportional Stratified Sampling helpers
# ------------------------------------------------------------------
def _proportional_stratified_sample(ds, total_n, seed, domain_shares, exact_targets=None):
    import random
    rng = random.Random(seed)

    domain_to_indices = defaultdict(list)
    for idx, domain in enumerate(ds["domain"]):
        if domain in domain_shares:
            domain_to_indices[domain].append(idx)

    if exact_targets is not None:
        allocations = exact_targets
    else:
        allocations = {}
        fractional = []
        for domain, share in domain_shares.items():
            exact = total_n * share
            floor_n = int(exact)
            allocations[domain] = floor_n
            fractional.append((exact - floor_n, domain))

        deficit = total_n - sum(allocations.values())
        fractional.sort(key=lambda x: x[0], reverse=True)
        for i in range(deficit):
            allocations[fractional[i % len(fractional)][1]] += 1

    selected = []
    for domain, n in allocations.items():
        indices = domain_to_indices.get(domain, [])
        if n > len(indices):
            n = len(indices)
        selected.extend(rng.sample(indices, n))

    shortfall = total_n - len(selected)
    if shortfall > 0:
        selected_set = set(selected)
        pool = [i for i in range(len(ds)) if i not in selected_set]
        if shortfall <= len(pool):
            selected.extend(rng.sample(pool, shortfall))
        else:
            selected.extend(pool)

    return ds.select(selected)

def _proportional_stratified_split(ds, val_size, seed, val_targets=None):
    import random
    rng = random.Random(seed)

    domain_to_indices = defaultdict(list)
    for idx, domain in enumerate(ds["domain"]):
        domain_to_indices[domain].append(idx)

    train_indices, val_indices = [], []

    if val_targets is not None:
        for domain, indices in domain_to_indices.items():
            n_val = val_targets.get(domain, 0)
            if len(indices) > 1 and n_val > 0:
                n_val = max(1, min(n_val, len(indices) - 1))
            else:
                n_val = 0
            val_set = set(rng.sample(indices, n_val))
            val_indices.extend(val_set)
            train_indices.extend([i for i in indices if i not in val_set])
    else:
        val_ratio = val_size / max(1, len(ds))
        for _domain, indices in domain_to_indices.items():
            n_val = int(round(len(indices) * val_ratio))
            if len(indices) > 1:
                n_val = max(1, min(n_val, len(indices) - 1))
            else:
                n_val = 0
            val_set = set(rng.sample(indices, n_val))
            val_indices.extend(val_set)
            train_indices.extend([i for i in indices if i not in val_set])

    while len(val_indices) < val_size and train_indices:
        val_indices.append(train_indices.pop())
    while len(val_indices) > val_size:
        train_indices.append(val_indices.pop())

    return ds.select(train_indices), ds.select(val_indices)

def _print_domain_counts(ds, label):
    counts = defaultdict(int)
    for d in ds["domain"]:
        counts[d] += 1
    total = sum(counts.values()) or 1
    parts = [f"{d}={c:,} ({100 * c / total:.2f}%)"
             for d, c in sorted(counts.items(), key=lambda x: -x[1])]
    print(f"  [{label}] {', '.join(parts)}")

def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _deduplicate(ds):
    seen = set()
    keep = []
    for i, text in enumerate(ds["text"]):
        h = _sha256(text)
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return ds.select(keep)

# ------------------------------------------------------------------
# Statistical Verifications
# ------------------------------------------------------------------
def _verify_domain_distribution(ds, domain_shares, label):
    try:
        from scipy.stats import chisquare
    except ImportError:
        print("  scipy not installed. Skipping Chi-Square test.")
        return
    counts = Counter(ds["domain"])
    total = len(ds)
    expected = [total * share for domain, share in domain_shares.items()]
    observed = [counts[domain] for domain in domain_shares.keys()]
    stat, p_value = chisquare(observed, f_exp=expected)
    print(f"  [{label}] Chi-Square: stat={stat:.2f}, p={p_value:.4f}")
    if p_value > CHI_SQUARE_P_THRESHOLD:
        print(f"  [{label}] Pass: p > {CHI_SQUARE_P_THRESHOLD}")
    else:
        print(f"  [{label}] WARNING: p <= {CHI_SQUARE_P_THRESHOLD}. Distribution may not match expected.")

def _calculate_kl_divergence(target_ds, full_ds, label, vocab_size=VOCAB_SIZE):
    import numpy as np
    print(f"  Calculating token frequencies for KL-Divergence ({label}) ...")
    tokenizer = _TOKENIZER
    
    p_counts = Counter()
    for text in target_ds["text"]:
        tokens = tokenizer.encode(text)
        p_counts.update(tokens)
        
    q_counts = Counter()
    for i in range(0, len(full_ds), 10000):
        batch_texts = full_ds[i:i+10000]["text"]
        for text in batch_texts:
            tokens = tokenizer.encode(text)
            q_counts.update(tokens)
            
    p_total = sum(p_counts.values())
    q_total = sum(q_counts.values())
    
    kl_div = 0.0
    for token_id in range(vocab_size):
        p_prob = p_counts[token_id] / p_total if p_total > 0 else 0
        q_prob = q_counts[token_id] / q_total if q_total > 0 else 0
        p_prob = max(p_prob, 1e-10)
        q_prob = max(q_prob, 1e-10)
        kl_div += p_prob * np.log(p_prob / q_prob)
        
    print(f"  [{label}] KL Divergence: {kl_div:.6f} nats")
    if kl_div < KL_DIVERGENCE_THRESHOLD:
        print(f"  [{label}] Pass: KL < {KL_DIVERGENCE_THRESHOLD}")
    else:
        print(f"  [{label}] WARNING: KL >= {KL_DIVERGENCE_THRESHOLD}")

def evaluate_zero_shot_with_bootstrap(eval_func, dataset, iterations=BOOTSTRAP_ITERATIONS):
    import numpy as np
    print(f"Evaluating zero-shot with {iterations}-iteration paired bootstrap ...")
    scores = np.array([eval_func(item) for item in dataset])
    n = len(scores)
    if n == 0:
        print("Empty dataset for evaluation.")
        return None
    bootstrap_means = []
    for _ in range(iterations):
        indices = np.random.randint(0, n, n)
        bootstrap_means.append(np.mean(scores[indices]))
    lower = np.percentile(bootstrap_means, 2.5)
    upper = np.percentile(bootstrap_means, 97.5)
    mean_score = np.mean(scores)
    print(f"Zero-Shot Score: {mean_score:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")
    return mean_score, lower, upper

# ------------------------------------------------------------------
# Dataset builders
# ------------------------------------------------------------------
def build_slimpajama(max_seq_len, stat_sizes, force_rebuild=False):
    params = {
        "max_seq_len":         max_seq_len,
        "pretrain_pool_size":  stat_sizes["PRETRAIN_POOL_SIZE"],
        "pretrain_seed":       PRETRAIN_SEED,
        "pretrain_size":       stat_sizes["PRETRAIN_SIZE"],
        "pretrain_val_size":   stat_sizes["PRETRAIN_VAL_SIZE"],
        "finetune_pool_size":  stat_sizes["FINETUNE_POOL_SIZE"],
        "finetune_seed":       FINETUNE_SEED,
        "finetune_size":       stat_sizes["FINETUNE_SIZE"],
        "finetune_val_size":   stat_sizes["FINETUNE_VAL_SIZE"],
        "domain_shares":       DOMAIN_SHARES,
    }
    keys = ["slimpajama_train", "slimpajama_val",
            "slimpajama_finetune", "slimpajama_finetune_val"]

    if (not _needs_rebuild("slimpajama", params, force_rebuild)
            and all(_dataset_saved(PATHS[k]) for k in keys)):
        print("SlimPajama splits already exist with matching params - skipping.")
        return

    if _needs_rebuild("slimpajama", params, force_rebuild):
        for k in keys:
            if os.path.isdir(PATHS[k]):
                shutil.rmtree(PATHS[k])

    print("Loading SlimPajama-6B train split ...")
    sp = _safe_load_dataset(
        "DKYoon/SlimPajama-6B", split="train", token=HF_TOKEN, streaming=False,
        download_mode="reuse_dataset_if_exists",
    )
    print(f"  Loaded {len(sp):,} rows")

    print("Extracting domain labels from meta['redpajama_set_name'] ...")
    sp = sp.map(
        lambda ex: {"domain": ex["meta"].get("redpajama_set_name", "Unknown")},
        desc="Extracting domain labels",
    )
    keep_cols = ["text", "domain"]
    drop_cols = [c for c in sp.column_names if c not in keep_cols]
    if drop_cols:
        sp = sp.remove_columns(drop_cols)

    print("Computing SHA-256 hashes and deduplicating ...")
    sp = _deduplicate(sp)
    print(f"  After dedup: {len(sp):,} rows")

    print(f"Drawing pretrain pool (n={stat_sizes['PRETRAIN_POOL_SIZE']:,}, seed={PRETRAIN_SEED}) ...")
    pretrain_pool_targets = {d: stat_sizes["PRETRAIN_DOMAIN_TARGETS"][d] + stat_sizes["PRETRAIN_VAL_DOMAIN_TARGETS"][d] for d in DOMAIN_SHARES}
    pretrain_pool = _proportional_stratified_sample(
        sp, stat_sizes["PRETRAIN_POOL_SIZE"], PRETRAIN_SEED, DOMAIN_SHARES, exact_targets=pretrain_pool_targets
    )
    print(f"  Pretrain pool: {len(pretrain_pool):,} rows")

    print("Splitting pretrain pool into pretrain / pretrain_val ...")
    pretrain, pretrain_val = _proportional_stratified_split(
        pretrain_pool, stat_sizes["PRETRAIN_VAL_SIZE"], seed=PRETRAIN_SEED, val_targets=stat_sizes["PRETRAIN_VAL_DOMAIN_TARGETS"]
    )
    print(f"  pretrain     : {len(pretrain):,} rows")
    print(f"  pretrain_val : {len(pretrain_val):,} rows")
    _print_domain_counts(pretrain, "pretrain")
    _print_domain_counts(pretrain_val, "pretrain_val")

    print("Removing pretrain-pool rows from dataset (zero overlap) ...")
    pretrain_hashes = set(_sha256(t) for t in pretrain_pool["text"])
    remaining_indices = [
        i for i, t in enumerate(sp["text"])
        if _sha256(t) not in pretrain_hashes
    ]
    remaining = sp.select(remaining_indices)
    print(f"  Remaining after pretrain-pool removal: {len(remaining):,} rows")

    print(f"Drawing finetune pool (n={stat_sizes['FINETUNE_POOL_SIZE']:,}, seed={FINETUNE_SEED}) ...")
    finetune_pool_targets = {d: stat_sizes["FINETUNE_DOMAIN_TARGETS"][d] + stat_sizes["FINETUNE_VAL_DOMAIN_TARGETS"][d] for d in DOMAIN_SHARES}
    finetune_pool = _proportional_stratified_sample(
        remaining, stat_sizes["FINETUNE_POOL_SIZE"], FINETUNE_SEED, DOMAIN_SHARES, exact_targets=finetune_pool_targets
    )
    print(f"  Finetune pool: {len(finetune_pool):,} rows")

    finetune, finetune_val = _proportional_stratified_split(
        finetune_pool, stat_sizes["FINETUNE_VAL_SIZE"], seed=FINETUNE_SEED, val_targets=stat_sizes["FINETUNE_VAL_DOMAIN_TARGETS"]
    )
    print(f"  finetune     : {len(finetune):,} rows")
    print(f"  finetune_val : {len(finetune_val):,} rows")
    _print_domain_counts(finetune, "finetune")
    _print_domain_counts(finetune_val, "finetune_val")

    print("Running statistical verification ...")
    _verify_domain_distribution(pretrain, DOMAIN_SHARES, "pretrain")
    _verify_domain_distribution(finetune, DOMAIN_SHARES, "finetune")
    _calculate_kl_divergence(pretrain, sp, "pretrain")
    _calculate_kl_divergence(finetune, sp, "finetune")

    for name, raw in [
        ("slimpajama_train",        pretrain),
        ("slimpajama_val",          pretrain_val),
        ("slimpajama_finetune",     finetune),
        ("slimpajama_finetune_val", finetune_val),
    ]:
        print(f"Tokenizing {name} ({len(raw):,}) ...")
        ds = Dataset.from_generator(
            lambda raw=raw: manual_tokenize(raw["text"], max_seq_len),
            features=LM_FEATURES,
        )
        ds.save_to_disk(PATHS[name])

    _record_built("slimpajama", params)


def build_minipile(max_seq_len, bench_sample_ratio=BENCH_SAMPLE_RATIO, force_rebuild=False):
    params = {
        "max_seq_len":        max_seq_len,
        "bench_sample_ratio": bench_sample_ratio,
        "minipile_total":     MINIPILE_TOTAL_ROWS,
    }

    if (not _needs_rebuild("minipile", params, force_rebuild)
            and _dataset_saved(PATHS["minipile"])):
        print("MiniPile already exists with matching params - skipping.")
        return

    if _needs_rebuild("minipile", params, force_rebuild):
        if os.path.isdir(PATHS["minipile"]):
            shutil.rmtree(PATHS["minipile"])

    print(f"Loading MiniPile test split (100% = {MINIPILE_TOTAL_ROWS:,} rows) ...")
    mp = _safe_load_dataset(
        "JeanKaddour/minipile", split="test",
        download_mode="force_redownload",
    )
    if bench_sample_ratio < 1.0:
        n = max(1, int(len(mp) * bench_sample_ratio))
        mp = mp.shuffle(seed=42).select(range(n))
    print(f"  Loaded {len(mp):,} MiniPile rows")

    if "meta" in mp.column_names:
        mp = mp.map(
            lambda ex: {"domain": ex["meta"].get("pile_set_name", "Unknown")},
            desc="Extracting MiniPile domain labels",
        )
    else:
        mp = mp.map(lambda ex: {"domain": "Unknown"}, desc="Adding placeholder domain")

    orig_counts = Counter(mp["domain"])
    orig_total = sum(orig_counts.values())
    MINIPILE_DOMAIN_SHARES = {d: c / orig_total for d, c in orig_counts.items()}

    print("Running statistical verification on MiniPile ...")
    _verify_domain_distribution(mp, MINIPILE_DOMAIN_SHARES, "minipile")
    
    sp_stream_subset = _safe_load_dataset("DKYoon/SlimPajama-6B", split="train", streaming=True, token=HF_TOKEN).take(50000)
    _calculate_kl_divergence(mp, sp_stream_subset, "minipile_vs_slimpajama")

    ds = Dataset.from_generator(
        lambda: manual_tokenize(mp["text"], max_seq_len),
        features=LM_FEATURES,
    )
    ds.save_to_disk(PATHS["minipile"])
    print(f"MiniPile ({len(ds):,}) saved to {PATHS['minipile']}")

    _record_built("minipile", params)


# Called from notebooks
def run_preprocessing(
    max_seq_len=None,
    minority_confidence=None,
    minority_margin_error=None,
    bench_sample_ratio=None,
    build_slimpajama_data=True,
    build_minipile_data=True,
    force_rebuild=False,
):
    """
    Unified entry point for preprocessing.
    """
    # Resolve defaults
    max_seq_len        = max_seq_len        if max_seq_len        is not None else MAX_SEQ_LEN
    minority_confidence = minority_confidence if minority_confidence is not None else MINORITY_CONFIDENCE
    minority_margin_error = minority_margin_error if minority_margin_error is not None else MINORITY_MARGIN_ERROR
    bench_sample_ratio = bench_sample_ratio if bench_sample_ratio is not None else BENCH_SAMPLE_RATIO

    # ------------------------------------------------------------------
    # Max Sequence Length Checking Code
    # ------------------------------------------------------------------
    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError(f"Invalid max_seq_len: {max_seq_len}. Must be a positive integer.")
    
    # Check tokenizer's max length limit (once loaded)
    _ensure_setup()
    if hasattr(_TOKENIZER, "model_max_length") and max_seq_len > _TOKENIZER.model_max_length:
        print(f"WARNING: Provided MAX_SEQ_LEN ({max_seq_len}) exceeds tokenizer max length ({_TOKENIZER.model_max_length}).")
        print("Truncation will be applied at the tokenizer's maximum limit.")
        
    # Dynamically calculate statistical sample sizes
    stat_sizes = get_stratified_sizes(minority_confidence, minority_margin_error)

    print("=" * 60)
    print("RUN PREPROCESSING (1D Domain Stratified Sampling)")
    print("=" * 60)
    print(f"  Max Sequence Length : {max_seq_len}")
    print(f"  Minority Confidence : {minority_confidence}")
    print(f"  Minority Margin Err : {minority_margin_error}")
    print(f"  Bench Sample Ratio  : {bench_sample_ratio}")
    print(f"  Cochran Minority N  : {stat_sizes['COCHRAN_MINORITY_N']:,} (StackExchange target)")
    print(f"  Pretrain pool size  : {stat_sizes['PRETRAIN_POOL_SIZE']:,} (seed={PRETRAIN_SEED})")
    print(f"  Pretrain size       : {stat_sizes['PRETRAIN_SIZE']:,}")
    print(f"  Pretrain val size   : {stat_sizes['PRETRAIN_VAL_SIZE']:,}")
    print(f"  Finetune pool size  : {stat_sizes['FINETUNE_POOL_SIZE']:,} (seed={FINETUNE_SEED})")
    print(f"  Finetune size       : {stat_sizes['FINETUNE_SIZE']:,}")
    print(f"  Finetune val size   : {stat_sizes['FINETUNE_VAL_SIZE']:,}")
    print(f"  Total sampled rows  : {stat_sizes['TOTAL_SAMPLED_ROWS']:,}")
    print("=" * 60)

    if build_slimpajama_data:
        build_slimpajama(max_seq_len=max_seq_len, stat_sizes=stat_sizes, force_rebuild=force_rebuild)
    if build_minipile_data:
        build_minipile(
            max_seq_len=max_seq_len,
            bench_sample_ratio=bench_sample_ratio,
            force_rebuild=force_rebuild,
        )

    print("\n" + "=" * 60)
    print("All Tokenized Datasets Ready")
    print("=" * 60)
    for name, path in PATHS.items():
        marker = "tokenizer_config.json" if name == "tokenizer" else "dataset_info.json"
        status = "OK" if os.path.exists(os.path.join(path, marker)) else "MISSING"
        print(f"  {status:7s} {name:24s} {path}")
    print("=" * 60)

if __name__ == "__main__":
    run_preprocessing()