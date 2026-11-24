"""Tunable constants and paths for the 1D domain-stratified preprocessing."""
import json
import os

from datasets import load_dataset_builder

from .statistical_sample import PAPER_DOMAIN_SHARES

HF_TOKEN             = "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"
TOKENIZER_NAME       = "deepseek-ai/deepseek-moe-16b-base"

# Default parameters (can be overridden from the notebook).
# MINORITY_MARGIN_ERROR is a RELATIVE margin: 0.02 = +-2% of the StackExchange
# share, which is the definition used in the paper.
MAX_SEQ_LEN              = 384
MINORITY_CONFIDENCE      = 0.95
MINORITY_MARGIN_ERROR    = 0.02

# Cochran's finite-population correction, n = n0 / (1 + (n0 - 1) / N).
# The parent split is finite (5.49M rows) and rows are drawn WITHOUT replacement,
# so the finite-population size is the statistically correct one:
#   infinite: n0 = z^2 (1-p) / (e^2 p) = 182,123
#   finite  : n  = n0 / (1 + (n0-1)/N)  = 176,274
# Set this to False to reproduce the published infinite-population figures.
USE_FINITE_POPULATION_CORRECTION = True

# Where the stratification weights come from.
#   "measured" - derive the domain shares from the downloaded corpus (default)
#   "paper"    - use the hard-coded SlimPajama Table-4 (token) shares
DOMAIN_SHARE_SOURCE      = "measured"

# Fraction of each *finished* SlimPajama split that is kept on disk.
# The stratified design is always drawn at full size first and this uniform
# random subsample is applied afterwards, so the four splits stay disjoint and
# keep their domain mixture in expectation.  1.0 keeps every designed row.
SAMPLE_RATIO       = 0.005
SAMPLE_SEED        = 44

# Fraction of the (decontaminated) MiniPile test split that is evaluated.
# 1.0 keeps the full published split; below 1.0 the draw is
# min(Cochran n, ratio * N) rows, i.e. a smoke test.
BENCH_SAMPLE_RATIO = 0.03

PRETRAIN_SEED      = 42
FINETUNE_SEED      = 43

# MiniPile test-split size.  ``load_dataset_builder`` exposes it on
# ``info.splits``; the old ``split=`` argument always raised and left the size
# at a hard-coded fallback.  An offline / Hub failure must not abort the import.
def _minipile_test_rows(default=10_000):
    try:
        split_info = load_dataset_builder("JeanKaddour/minipile").info.splits.get("test")
    except Exception as exc:
        print(f"Warning: could not read MiniPile metadata ({exc}); using {default:,}.")
        return default
    if split_info is None or split_info.num_examples is None:
        print(f"Warning: MiniPile metadata has no usable 'test' split; using {default:,}.")
        return default
    return split_info.num_examples


MINIPILE_TOTAL_ROWS    = _minipile_test_rows()
print(f"MiniPile test size: {MINIPILE_TOTAL_ROWS:,} rows")

BOOTSTRAP_ITERATIONS    = 1_000

# Statistical verification thresholds
CHI_SQUARE_P_THRESHOLD  = 0.95
KL_DIVERGENCE_THRESHOLD = 0.001
VOCAB_SIZE              = 102400

# Token-frequency estimation is done on a bounded number of documents so the
# KL check finishes in minutes instead of tokenising the full 5.49M-row corpus.
KL_TARGET_MAX_DOCS      = 50_000
KL_REFERENCE_MAX_DOCS   = 100_000

DATASET_ROOT = "/content/deepseekmoe_dynamic_routing_algorithms/dataset"
REPO_ROOT    = "/content/deepseekmoe_dynamic_routing_algorithms"

DEFAULT_TOKENIZER_PATH  = os.path.join(DATASET_ROOT, "tokenizer")
DEFAULT_TRAIN_PATH      = os.path.join(DATASET_ROOT, "slimpajama_train")
DEFAULT_FINE_PATH       = os.path.join(DATASET_ROOT, "slimpajama_finetune")
DEFAULT_VAL_PATH        = os.path.join(DATASET_ROOT, "slimpajama_val")
DEFAULT_FINE_VAL_PATH   = os.path.join(DATASET_ROOT, "slimpajama_finetune_val")
DEFAULT_MINIPILE_PATH   = os.path.join(DATASET_ROOT, "minipile")

PATHS = {
    "tokenizer":               DEFAULT_TOKENIZER_PATH,
    "slimpajama_train":        DEFAULT_TRAIN_PATH,
    "slimpajama_finetune":     DEFAULT_FINE_PATH,
    "slimpajama_val":          DEFAULT_VAL_PATH,
    "slimpajama_finetune_val": DEFAULT_FINE_VAL_PATH,
    "minipile":                DEFAULT_MINIPILE_PATH,
}

META_FILE           = os.path.join(DATASET_ROOT, "_preprocessing_meta.json")

# Persistent 64-bit SHA-256 index of the deduplicated SlimPajama corpus, used to
# cross-decontaminate MiniPile without hashing 5.49M documents a second time.
DEDUP_HASH_FILE     = os.path.join(DATASET_ROOT, "_slimpajama_dedup_hashes.npy")

# Cache of the domain shares measured from the corpus, so a re-run can decide
# whether the existing splits still match without reloading 5.49M documents.
DOMAIN_SHARES_FILE  = os.path.join(DATASET_ROOT, "_domain_shares.json")


def _resolve_domain_shares():
    """Stratification weights, measured from the corpus when available.

    ``build_slimpajama`` writes the measured document shares to
    ``DOMAIN_SHARES_FILE``; importing the package after a run therefore exposes
    the *measured* weights as ``DOMAIN_SHARES``.  Before the corpus has ever
    been counted (fresh checkout) the published Table-4 values are used purely
    as a placeholder.

    Returns ``(shares, origin)``.
    """
    try:
        with open(DOMAIN_SHARES_FILE, "r") as f:
            payload = json.load(f)
        shares = {str(d): float(w)
                  for d, w in (payload.get("shares") or {}).items()
                  if float(w) > 0}
        total = sum(shares.values())
        if shares and total > 0:
            return {d: w / total for d, w in shares.items()}, "measured"
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"Warning: ignoring cached domain shares ({exc}).")
    return dict(PAPER_DOMAIN_SHARES), "paper (Table 4 fallback)"


DOMAIN_SHARES, DOMAIN_SHARES_ORIGIN = _resolve_domain_shares()

