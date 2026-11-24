from .config import (
    HF_TOKEN, TOKENIZER_NAME, MAX_SEQ_LEN, BENCH_SAMPLE_RATIO,
    MINORITY_CONFIDENCE, MINORITY_MARGIN_ERROR, DOMAIN_SHARES,
    PAPER_DOMAIN_SHARES, DOMAIN_SHARE_SOURCE, DOMAIN_SHARES_FILE,
    DOMAIN_SHARES_ORIGIN, SAMPLE_RATIO, SAMPLE_SEED,
    USE_FINITE_POPULATION_CORRECTION,
    PRETRAIN_SEED, FINETUNE_SEED,
    MINIPILE_TOTAL_ROWS, BOOTSTRAP_ITERATIONS,
    CHI_SQUARE_P_THRESHOLD, KL_DIVERGENCE_THRESHOLD, VOCAB_SIZE,
    KL_TARGET_MAX_DOCS, KL_REFERENCE_MAX_DOCS, DEDUP_HASH_FILE,
    DATASET_ROOT, REPO_ROOT, META_FILE,
    DEFAULT_TOKENIZER_PATH, DEFAULT_TRAIN_PATH, DEFAULT_FINE_PATH,
    DEFAULT_VAL_PATH, DEFAULT_FINE_VAL_PATH, DEFAULT_MINIPILE_PATH,
    PATHS as DATASET_PATHS,
)
PATHS = DATASET_PATHS
# `source.data_preprocessing` is the package; the builder module inside it is
# `preprocessing.py`, which is what the notebook command imports.
from .preprocessing import (
    build_minipile, build_slimpajama, evaluate_zero_shot_with_bootstrap,
    manual_tokenize, run_preprocessing,
)
from .statistical_sample import (
    MINORITY_DOMAIN, benchmark_sample_size, domain_shares_from_counts,
    finite_population_size, get_stratified_sizes, measure_domain_shares,
    relative_half_width, subsample_size, z_value,
)

__all__ = [
    "HF_TOKEN", "TOKENIZER_NAME", "MAX_SEQ_LEN", "BENCH_SAMPLE_RATIO",
    "MINORITY_CONFIDENCE", "MINORITY_MARGIN_ERROR", "DOMAIN_SHARES",
    "PAPER_DOMAIN_SHARES", "DOMAIN_SHARE_SOURCE", "DOMAIN_SHARES_FILE",
    "DOMAIN_SHARES_ORIGIN", "SAMPLE_RATIO", "SAMPLE_SEED",
    "USE_FINITE_POPULATION_CORRECTION",
    "PRETRAIN_SEED", "FINETUNE_SEED",
    "MINIPILE_TOTAL_ROWS", "BOOTSTRAP_ITERATIONS",
    "CHI_SQUARE_P_THRESHOLD", "KL_DIVERGENCE_THRESHOLD", "VOCAB_SIZE",
    "KL_TARGET_MAX_DOCS", "KL_REFERENCE_MAX_DOCS", "DEDUP_HASH_FILE",
    "DATASET_ROOT", "REPO_ROOT", "META_FILE",
    "DEFAULT_TOKENIZER_PATH", "DEFAULT_TRAIN_PATH", "DEFAULT_FINE_PATH",
    "DEFAULT_VAL_PATH", "DEFAULT_FINE_VAL_PATH", "DEFAULT_MINIPILE_PATH",
    "DATASET_PATHS", "PATHS", "run_preprocessing", "build_slimpajama", "build_minipile",
    "manual_tokenize", "evaluate_zero_shot_with_bootstrap",
    "MINORITY_DOMAIN",
    "domain_shares_from_counts", "measure_domain_shares", "get_stratified_sizes",
    "benchmark_sample_size", "relative_half_width", "subsample_size",
    "finite_population_size", "z_value",
]
