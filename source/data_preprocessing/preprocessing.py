# ============================================================
# source/data_preprocessing/prepare_datasets.py
# ============================================================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
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
    SAMPLE_RATIO, BENCH_SAMPLE_RATIO, AMBIGQA_MAX_EXAMPLES,
    DATASET_ROOT, PATHS, DEFAULT_TOKENIZER_PATH,
)

disable_progress_bar()

# --- Auth ---
login(token=HF_TOKEN)

# --- Tokenizer (single save) ---
os.makedirs(DEFAULT_TOKENIZER_PATH, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME, trust_remote_code=True, use_fast=True,
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.save_pretrained(DEFAULT_TOKENIZER_PATH)
print(f"Tokenizer saved to {DEFAULT_TOKENIZER_PATH}")


def _dataset_saved(path):
    return any(
        os.path.exists(os.path.join(path, m))
        for m in ("dataset_info.json", "state.json")
    )


# --- Helpers ---
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


LM_FEATURES = Features({
    "input_ids":      Sequence(Value("int32")),
    "attention_mask": Sequence(Value("int8")),
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


# --- SlimPajama splits ---
def build_slimpajama():
    keys = ["slimpajama_train", "slimpajama_finetune", "slimpajama_val"]
    if all(_dataset_saved(PATHS[k]) for k in keys):
        print("SlimPajama splits already exist - skipping.")
        return

    print("Loading SlimPajama-6B ...")
    sp = load_dataset(
        "DKYoon/SlimPajama-6B", token=HF_TOKEN, streaming=False,
        download_mode="reuse_dataset_if_exists",
    )
    drop = [c for c in ["__index_level_0__", "meta"]
            if c in sp["train"].column_names]
    if drop:
        sp = sp.remove_columns(drop)

    train_n = max(1, int(len(sp["train"])      * SAMPLE_RATIO))
    val_n   = max(1, int(len(sp["validation"]) * 0.05))

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


# --- MiniPile ---
def build_minipile():
    if _dataset_saved(PATHS["minipile"]):
        print("MiniPile already exists - skipping.")
        return
    print("Loading MiniPile ...")
    mp = safe_sample(load_dataset("JeanKaddour/minipile", split="test"),
                     BENCH_SAMPLE_RATIO)
    ds = Dataset.from_generator(
        lambda: manual_tokenize(mp["text"], MAX_SEQ_LEN),
        features=LM_FEATURES,
    )
    ds.save_to_disk(PATHS["minipile"])
    print(f"MiniPile ({len(ds)}) saved to {PATHS['minipile']}")


# --- ARC ---
def build_arc():
    if _dataset_saved(PATHS["arc"]):
        print("ARC already exists - skipping.")
        return
    print("Loading ARC-Challenge ...")
    arc = safe_sample(
        load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"),
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


# --- RACE ---
def build_race():
    if _dataset_saved(PATHS["race"]):
        print("RACE already exists - skipping.")
        return
    print("Loading RACE ...")
    race = safe_sample(
        load_dataset("ehovy/race", "all", split="test"),
        BENCH_SAMPLE_RATIO,
    )

    def fmt(ex):
        o = list(ex["options"])
        prompt = (
            f"Article: {ex['article']}\n"
            f"Question: {ex['question']}\n"
            f"Options:\nA) {o[0]}\nB) {o[1]}\nC) {o[2]}\nD) {o[3]}\n"
            f"Answer:"
        )
        return {"prompt": prompt,
                "choices": o,
                "choice_labels": ["A", "B", "C", "D"],
                "label": ex["answer"]}

    ds = race.map(fmt, remove_columns=race.column_names,
                  features=MC_FEATURES, num_proc=1)
    ds.save_to_disk(PATHS["race"])
    print(f"RACE ({len(ds)}) saved to {PATHS['race']}")


# --- TriviaQA ---
def build_triviaqa():
    if _dataset_saved(PATHS["triviaqa"]):
        print("TriviaQA already exists - skipping.")
        return
    print("Loading TriviaQA (mandarjoshi/trivia_qa, rc.nocontext, validation) ...")
    tr = safe_sample(
        load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation"),
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


# --- AmbigQA ---
def _flatten_answers(value):
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_answers(item))
        return out
    return []


def _collect_ambigqa_answers(example):
    annotations = example["annotations"]
    answers = set()
    for i, ann_type in enumerate(annotations["type"]):
        if ann_type == "singleAnswer":
            answers.update(_flatten_answers(annotations["answer"][i]))
        else:
            qa_pair = annotations["qaPairs"][i]
            answers.update(_flatten_answers(qa_pair["answer"]))
    answers.update(_flatten_answers(example["nq_answer"]))
    return sorted(answers) or [""]


def build_ambigqa():
    if _dataset_saved(PATHS["ambigqa"]):
        print("AmbigQA already exists - skipping.")
        return

    print("Loading AmbigQA (sewon/ambig_qa, full, validation) ...")
    full = load_dataset("sewon/ambig_qa", "full", split="validation")
    if len(full) > AMBIGQA_MAX_EXAMPLES:
        full = full.shuffle(seed=42).select(range(AMBIGQA_MAX_EXAMPLES))
    print(f"  Using {len(full)} validation examples")

    records = [
        {
            "prompt":  f"Question: {ex['question'].strip()}\nAnswer:",
            "answers": _collect_ambigqa_answers(ex),
        }
        for ex in full
    ]
    ds = Dataset.from_list(records, features=QA_FEATURES)
    ds.save_to_disk(PATHS["ambigqa"])
    print(f"AmbigQA ({len(ds)}) saved to {PATHS['ambigqa']}")


if __name__ == "__main__":
    build_slimpajama()
    build_minipile()
    build_arc()
    build_race()
    build_triviaqa()
    build_ambigqa()

    print("\n" + "=" * 60)
    print("All tokenized datasets saved under:", DATASET_ROOT)
    print("=" * 60)
    for name, path in PATHS.items():
        marker = "tokenizer_config.json" if name == "tokenizer" else "dataset_info.json"
        status = "OK" if os.path.exists(os.path.join(path, marker)) else "MISSING"
        print(f"  {status:7s} {name:24s} {path}")