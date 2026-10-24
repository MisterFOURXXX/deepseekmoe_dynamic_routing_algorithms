import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
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

from config import (
    HF_TOKEN, TOKENIZER_NAME, MAX_SEQ_LEN,
    SAMPLE_RATIO, BENCH_SAMPLE_RATIO,
    DATASET_ROOT, PATHS, DEFAULT_TOKENIZER_PATH, REPO_ROOT
)

# Redirect HF cache
os.environ["HF_DATASETS_CACHE"]                 = REPO_ROOT + "/hf_cache"

disable_progress_bar()

# Auth
login(token=HF_TOKEN)

# Tokenizer 
os.makedirs(DEFAULT_TOKENIZER_PATH, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME, trust_remote_code=True, use_fast=True,
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.save_pretrained(DEFAULT_TOKENIZER_PATH)
print(f"Tokenizer saved to {DEFAULT_TOKENIZER_PATH}")


# Corruption recovery helpers
def _dataset_saved(path):
    """
    True only if a saved dataset exists AND can be loaded.
    If it exists but fails to load (corrupted features), it is
    deleted so it will be re-processed.
    """
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
    """
    Delete the HF datasets **builder** cache for a specific
    dataset / config. This is separate from save_to_disk() and is
    where `DatasetBuilder.__init__` reads dataset_info.json.
    """
    if not dataset_name or "/" not in dataset_name:
        return
    org, _, name = dataset_name.partition("/")
    base = os.path.join(
        os.environ["HF_DATASETS_CACHE"],
        f"{org}___{name}",
    )
    if config_name:
        base = os.path.join(base, config_name)
    if os.path.isdir(base):
        shutil.rmtree(base)
        print(f"  Cleared HF builder cache: {base}")


def _safe_load_dataset(*args, **kwargs):
    """
    Wrapper around load_dataset() that auto-recovers from the
    `TypeError: must be called with a dataclass type or instance`
    error caused by a corrupted HF builder cache.
    """
    try:
        return load_dataset(*args, **kwargs)
    except TypeError as e:
        msg = str(e).lower()
        if "dataclass" not in msg:
            raise
        dataset_name = args[0] if args else kwargs.get("path")
        config_name  = args[1] if len(args) > 1 else kwargs.get("name")
        print(f"  Detected corrupted builder cache for "
              f"{dataset_name}/{config_name}. "
              f"Clearing and retrying ...")
        _clear_builder_cache(dataset_name, config_name)
        return load_dataset(*args, **kwargs)


# Helpers
def safe_sample(ds, ratio, min_n=1):
    n = max(min_n, int(len(ds) * ratio))
    return ds.shuffle(seed=42).select(range(min(n, len(ds))))


def manual_tokenize(text_iter, max_len, batch_size=8192):
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


# int32 for BOTH fields (int8 does not round-trip in datasets==2.21.0)
LM_FEATURES = Features({
    "input_ids":      Sequence(Value("int32")),
    "attention_mask": Sequence(Value("int32")),
})
MC_FEATURES = Features({
    "prompt":        Value("string"),
    "choices":       Sequence(Value("string")),
    "choice_labels": Sequence(Value("string")),
    "label":         Value("string"),
})
QA_FEATURES = Features({
    "prompt":  Value("string"),
    "answers": Sequence(Value("string")),
})


# SlimPajama splits
def build_slimpajama():
    keys = ["slimpajama_train", "slimpajama_finetune", "slimpajama_val"]
    if all(_dataset_saved(PATHS[k]) for k in keys):
        print("SlimPajama splits already exist - skipping.")
        return

    print("Loading SlimPajama-6B ...")
    sp = _safe_load_dataset(
        "DKYoon/SlimPajama-6B", token=HF_TOKEN, streaming=False,
        download_mode="reuse_dataset_if_exists",
    )
    drop = [c for c in ["__index_level_0__", "meta"]
            if c in sp["train"].column_names]
    if drop:
        sp = sp.remove_columns(drop)

    train_n = max(1, int(len(sp["train"])      * SAMPLE_RATIO))
    val_n   = max(1, int(len(sp["validation"]) * 0.1))

    train_sub = sp["train"].shuffle(seed=42).select(range(train_n))
    val_sub   = sp["validation"].shuffle(seed=42).select(range(val_n))

    split_idx    = int(len(train_sub) * 0.6)
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
            lambda raw=raw: manual_tokenize(raw["text"], MAX_SEQ_LEN),
            features=LM_FEATURES,
        )
        ds.save_to_disk(PATHS[name])
        print(f"  Saved to {PATHS[name]}")


# MiniPile
def build_minipile():
    if _dataset_saved(PATHS["minipile"]):
        print("MiniPile already exists - skipping.")
        return
    print("Loading MiniPile ...")
    mp = safe_sample(
        _safe_load_dataset(
            "JeanKaddour/minipile", split="test",
            download_mode="force_redownload",
        ),
        BENCH_SAMPLE_RATIO,
    )
    ds = Dataset.from_generator(
        lambda: manual_tokenize(mp["text"], MAX_SEQ_LEN),
        features=LM_FEATURES,
    )
    ds.save_to_disk(PATHS["minipile"])
    print(f"MiniPile ({len(ds)}) saved to {PATHS['minipile']}")


# ARC
def build_arc():
    if _dataset_saved(PATHS["arc"]):
        print("ARC already exists - skipping.")
        return
    print("Loading ARC-Challenge ...")
    arc = safe_sample(
        _safe_load_dataset(
            "allenai/ai2_arc", "ARC-Challenge", split="test",
            download_mode="force_redownload",
        ),
        BENCH_SAMPLE_RATIO,
    )

    def fmt(ex):
        labels = list(ex["choices"]["label"])
        texts  = list(ex["choices"]["text"])
        prompt = f"Question: {ex['question']}\nChoices:\n"
        for lab, txt in zip(labels, texts):
            prompt += f"{lab}. {txt}\n"
        return {"prompt": prompt + "Answer:",
                "choices": texts,
                "choice_labels": labels,
                "label": ex["answerKey"]}

    ds = arc.map(fmt, remove_columns=arc.column_names,
                 features=MC_FEATURES, num_proc=1)
    ds.save_to_disk(PATHS["arc"])
    print(f"ARC ({len(ds)}) saved to {PATHS['arc']}")


# TriviaQA
def build_triviaqa():
    if _dataset_saved(PATHS["triviaqa"]):
        print("TriviaQA already exists - skipping.")
        return
    print("Loading TriviaQA (mandarjoshi/trivia_qa, rc.nocontext, validation) ...")
    tr = safe_sample(
        _safe_load_dataset(
            "mandarjoshi/trivia_qa", "rc.nocontext", split="validation",
            download_mode="force_redownload",
        ),
        BENCH_SAMPLE_RATIO,
    )

    def fmt(ex):
        raw = list(ex["answer"].get("aliases", [])) + [ex["answer"].get("value", "")]
        answers = list({a.strip() for a in raw if a and a.strip()}) or [""]
        return {"prompt": f"Question: {ex['question']}\nAnswer:",
                "answers": answers}

    ds = tr.map(fmt, remove_columns=tr.column_names,
                features=QA_FEATURES, num_proc=1)
    ds.save_to_disk(PATHS["triviaqa"])
    print(f"TriviaQA ({len(ds)}) saved to {PATHS['triviaqa']}")


if __name__ == "__main__":
    build_slimpajama()
    build_minipile()
    build_arc()
    build_triviaqa()

    print("\n" + "=" * 60)
    print("All tokenized datasets saved under:", DATASET_ROOT)
    print("=" * 60)
    for name, path in PATHS.items():
        marker = "tokenizer_config.json" if name == "tokenizer" else "dataset_info.json"
        status = "OK" if os.path.exists(os.path.join(path, marker)) else "MISSING"
        print(f"  {status:7s} {name:24s} {path}")