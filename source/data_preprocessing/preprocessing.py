import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import json
import shutil

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
    SAMPLE_RATIO, BENCH_SAMPLE_RATIO,
    VALIDATION_RATIO, TRAIN_SPLIT_RATIO,
    DATASET_ROOT, PATHS, DEFAULT_TOKENIZER_PATH, REPO_ROOT,
    META_FILE,
)

os.environ["HF_DATASETS_CACHE"] = REPO_ROOT + "/hf_cache"
disable_progress_bar()


# ------------------------------------------------------------------
# Lazy setup (avoids running login/tokenizer at import time)
# ------------------------------------------------------------------
_TOKENIZER = None


def _ensure_setup():
    """Idempotent setup: HF login + tokenizer. Called by run_preprocessing()."""
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


# ------------------------------------------------------------------
# Meta tracking (auto-rebuild when parameters change)
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# Corruption recovery helpers
# ------------------------------------------------------------------
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
        print(f"  WARNING: corrupted saved dataset at {path}: "
              f"{type(e).__name__}: {e}")
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
        print(f"  Cleared HF builder cache: {base}")


def _safe_load_dataset(*args, **kwargs):
    try:
        return load_dataset(*args, **kwargs)
    except TypeError as e:
        msg = str(e).lower()
        if "dataclass" not in msg:
            raise
        dataset_name = args[0] if args else kwargs.get("path")
        config_name  = args[1] if len(args) > 1 else kwargs.get("name")
        print(f"  Detected corrupted builder cache for "
              f"{dataset_name}/{config_name}. Clearing and retrying ...")
        _clear_builder_cache(dataset_name, config_name)
        return load_dataset(*args, **kwargs)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def safe_sample(ds, ratio, min_n=1):
    n = max(min_n, int(len(ds) * ratio))
    return ds.shuffle(seed=42).select(range(min(n, len(ds))))


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
# Dataset builders (accept runtime parameters)
# ------------------------------------------------------------------
def build_slimpajama(max_seq_len, sample_ratio, validation_ratio,
                     train_split_ratio, force_rebuild=False):
    params = {
        "max_seq_len":       max_seq_len,
        "sample_ratio":      sample_ratio,
        "validation_ratio":  validation_ratio,
        "train_split_ratio": train_split_ratio,
    }
    keys = ["slimpajama_train", "slimpajama_finetune", "slimpajama_val"]

    if (not _needs_rebuild("slimpajama", params, force_rebuild)
            and all(_dataset_saved(PATHS[k]) for k in keys)):
        print(f"SlimPajama splits already exist with matching params - skipping.")
        print(f"  Current params: {params}")
        return

    if _needs_rebuild("slimpajama", params, force_rebuild):
        for k in keys:
            if os.path.isdir(PATHS[k]):
                shutil.rmtree(PATHS[k])

    print(f"Loading SlimPajama-6B with params: {params}")
    sp = _safe_load_dataset(
        "DKYoon/SlimPajama-6B", token=HF_TOKEN, streaming=False,
        download_mode="reuse_dataset_if_exists",
    )
    drop = [c for c in ["__index_level_0__", "meta"]
            if c in sp["train"].column_names]
    if drop:
        sp = sp.remove_columns(drop)

    train_n = max(1, int(len(sp["train"])      * sample_ratio))
    val_n   = max(1, int(len(sp["validation"]) * validation_ratio))

    train_sub = sp["train"].shuffle(seed=42).select(range(train_n))
    val_sub   = sp["validation"].shuffle(seed=42).select(range(val_n))

    split_idx    = int(len(train_sub) * train_split_ratio)
    train_raw    = train_sub.select(range(split_idx))
    finetune_raw = train_sub.select(range(split_idx, len(train_sub)))

    print(f"  Train    : {len(train_raw):,}")
    print(f"  Finetune : {len(finetune_raw):,}")
    print(f"  Val      : {len(val_sub):,}")

    for name, raw in [("slimpajama_train",    train_raw),
                      ("slimpajama_finetune", finetune_raw),
                      ("slimpajama_val",      val_sub)]:
        print(f"Tokenizing {name} ({len(raw)}) ...")
        ds = Dataset.from_generator(
            lambda raw=raw: manual_tokenize(raw["text"], max_seq_len),
            features=LM_FEATURES,
        )
        ds.save_to_disk(PATHS[name])
        print(f"  Saved to {PATHS[name]}")

    _record_built("slimpajama", params)


def build_minipile(max_seq_len, bench_sample_ratio, force_rebuild=False):
    params = {
        "max_seq_len":       max_seq_len,
        "bench_sample_ratio": bench_sample_ratio,
    }

    if (not _needs_rebuild("minipile", params, force_rebuild)
            and _dataset_saved(PATHS["minipile"])):
        print(f"MiniPile already exists with matching params - skipping.")
        print(f"  Current params: {params}")
        return

    if _needs_rebuild("minipile", params, force_rebuild):
        if os.path.isdir(PATHS["minipile"]):
            shutil.rmtree(PATHS["minipile"])

    print(f"Loading MiniPile with params: {params}")
    mp = safe_sample(
        _safe_load_dataset(
            "JeanKaddour/minipile", split="test",
            download_mode="force_redownload",
        ),
        bench_sample_ratio,
    )
    ds = Dataset.from_generator(
        lambda: manual_tokenize(mp["text"], max_seq_len),
        features=LM_FEATURES,
    )
    ds.save_to_disk(PATHS["minipile"])
    print(f"MiniPile ({len(ds)}) saved to {PATHS['minipile']}")

    _record_built("minipile", params)


# Called from notebooks
def run_preprocessing(
    max_seq_len=None,
    sample_ratio=None,
    validation_ratio=None,
    train_split_ratio=None,
    bench_sample_ratio=None,
    build_slimpajama_data=True,
    build_minipile_data=True,
    force_rebuild=False,
):
    """
    Unified entry point for preprocessing.

    Parameters can be overridden from the notebook. If the override differs
    from what was used previously (tracked in `_preprocessing_meta.json`),
    the affected datasets are rebuilt automatically.

    Args:
        max_seq_len:        Override for MAX_SEQ_LEN.
        sample_ratio:       Override for SAMPLE_RATIO (SlimPajama subset).
        validation_ratio:   Override for VALIDATION_RATIO (HF val split fraction).
        train_split_ratio:  Override for TRAIN_SPLIT_RATIO (train/finetune split).
        bench_sample_ratio: Override for BENCH_SAMPLE_RATIO (MiniPile subset).
        build_slimpajama_data: If True, build SlimPajama train/finetune/val splits.
        build_minipile_data:   If True, build MiniPile benchmark.
        force_rebuild:      If True, ignore meta and always rebuild.
    """
    # Resolve defaults
    max_seq_len        = max_seq_len        if max_seq_len        is not None else MAX_SEQ_LEN
    sample_ratio       = sample_ratio       if sample_ratio       is not None else SAMPLE_RATIO
    validation_ratio   = validation_ratio   if validation_ratio   is not None else VALIDATION_RATIO
    train_split_ratio  = train_split_ratio  if train_split_ratio  is not None else TRAIN_SPLIT_RATIO
    bench_sample_ratio = bench_sample_ratio if bench_sample_ratio is not None else BENCH_SAMPLE_RATIO

    print("=" * 60)
    print("RUN PREPROCESSING")
    print("=" * 60)
    print(f"  Max Sequence Length        : {max_seq_len}")
    print(f"  Sample Ratio               : {sample_ratio}")
    print(f"  Validation Ratio           : {validation_ratio}")
    print(f"  Train Split Ratio          : {train_split_ratio}")
    print(f"  Bench Sample Ratio         : {bench_sample_ratio}")
    print(f"  Build SlimPajama train/fine tune/val splits          : {build_slimpajama_data}")
    print(f"  Build MiniPile Benchmark     : {build_minipile_data}")
    print(f"  Rebuild Preprocess Data      : {force_rebuild}")
    print("=" * 60)

    _ensure_setup()

    if build_slimpajama_data:
        build_slimpajama(
            max_seq_len=max_seq_len,
            sample_ratio=sample_ratio,
            validation_ratio=validation_ratio,
            train_split_ratio=train_split_ratio,
            force_rebuild=force_rebuild,
        )
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