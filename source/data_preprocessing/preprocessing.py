"""SlimPajama / MiniPile 1D domain-stratified sampling and tokenisation.

Why this version is fast:

* Domain labels come from Arrow (``pc.struct_field`` + ``pc.dictionary_encode``)
  as integer codes, so the 5.49M Python strings the old code materialised never
  exist.
* Documents are hashed into a single int64 column (44 MB) instead of rewriting
  the ~12 GB text column, and deduplication keeps an index array instead of
  copying the dataset.
* The finetune pool is carved out with boolean masks on the already-known codes,
  so no second factorisation of 5.3M domain strings is needed.
* KL diagnostics count the tokens of the splits that were actually saved with an
  Arrow scan (``pc.value_counts``); only the reference corpus is tokenised.
* Every split is a single ``take`` of the original memory-mapped table.
"""
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import hashlib
import json
import os
import shutil
from collections import Counter
from functools import partial

os.environ.setdefault("HF_DATASETS_DISABLE_DILL", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import pandas as pd
import pyarrow.compute as pc
from datasets import Dataset, Features, Sequence, Value, load_dataset
from datasets.utils.logging import disable_progress_bar
from huggingface_hub import login
from scipy.stats import chisquare
from transformers import AutoTokenizer

from .config import (
    BENCH_SAMPLE_RATIO, BOOTSTRAP_ITERATIONS, CHI_SQUARE_P_THRESHOLD,
    DEDUP_HASH_FILE, DEFAULT_TOKENIZER_PATH, DOMAIN_SHARE_SOURCE,
    DOMAIN_SHARES_FILE, FINETUNE_SEED, HF_TOKEN, KL_DIVERGENCE_THRESHOLD,
    KL_REFERENCE_MAX_DOCS, KL_TARGET_MAX_DOCS, MAX_SEQ_LEN, META_FILE,
    MINIPILE_TOTAL_ROWS, MINORITY_CONFIDENCE, MINORITY_MARGIN_ERROR, PATHS,
    PRETRAIN_SEED, REPO_ROOT, SAMPLE_RATIO, SAMPLE_SEED, TOKENIZER_NAME,
    USE_FINITE_POPULATION_CORRECTION, VOCAB_SIZE,
)
from .statistical_sample import (
    PAPER_DOMAIN_SHARES, allocate_with_capacity, benchmark_sample_size,
    domain_shares_from_counts, get_stratified_sizes, largest_remainder,
    relative_half_width, subsample_size,
)

os.environ["HF_DATASETS_CACHE"] = REPO_ROOT + "/hf_cache"
disable_progress_bar()

NUM_PROC = max(1, min(8, os.cpu_count() or 1))

LM_FEATURES = Features({
    "input_ids":      Sequence(Value("int32")),
    "attention_mask": Sequence(Value("int32")),
})

HASH_FEATURES = Features({"h": Value("int64")})
DOMAIN_FEATURES = Features({"domain": Value("string")})

TOKENIZE_BATCH = 4096
HASH_BATCH = 4096

_TOKENIZER = None


# ------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------
def _ensure_setup():
    """Load the tokenizer once; reuse the local copy when it exists.

    Reusing the saved copy also makes the function safe to call from
    ``Dataset.map(num_proc=...)`` workers (no concurrent logins / re-saves).
    """
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER

    local_config = os.path.join(DEFAULT_TOKENIZER_PATH, "tokenizer_config.json")
    if os.path.exists(local_config):
        _TOKENIZER = AutoTokenizer.from_pretrained(
            DEFAULT_TOKENIZER_PATH, trust_remote_code=True, use_fast=True,
        )
    else:
        login(token=HF_TOKEN)
        os.makedirs(DEFAULT_TOKENIZER_PATH, exist_ok=True)
        _TOKENIZER = AutoTokenizer.from_pretrained(
            TOKENIZER_NAME, trust_remote_code=True, use_fast=True,
        )
        _TOKENIZER.save_pretrained(DEFAULT_TOKENIZER_PATH)
        print(f"Tokenizer saved to {DEFAULT_TOKENIZER_PATH}")

    _TOKENIZER.pad_token = _TOKENIZER.eos_token
    _TOKENIZER.pad_token_id = _TOKENIZER.eos_token_id
    return _TOKENIZER

# ------------------------------------------------------------------
# Bookkeeping
# ------------------------------------------------------------------
def _load_meta():
    if not os.path.exists(META_FILE):
        return {}
    with open(META_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"WARNING: unreadable {META_FILE}; rebuilding from scratch.")
            return {}


def _save_meta(meta):
    os.makedirs(os.path.dirname(META_FILE), exist_ok=True)
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)


def _params_match(stored, params, defaults=None):
    """True when every requested parameter is present and equal in ``stored``.

    Only the keys in ``params`` are compared, so a builder can record extra
    diagnostics (realised row counts, decontamination losses) without breaking
    the cache check.  ``defaults`` gives the value to assume when an older meta
    file predates a newly added parameter.
    """
    if not isinstance(stored, dict):
        return False
    defaults = defaults or {}
    for key, value in params.items():
        if stored.get(key, defaults.get(key)) != value:
            return False
    return True


def _needs_rebuild(key, params, force_rebuild, defaults=None):
    if force_rebuild:
        return True
    return not _params_match(_load_meta().get(key), params, defaults)


def _record_built(key, params):
    meta = _load_meta()
    meta[key] = params
    _save_meta(meta)


def _dataset_saved(path):
    """True if ``path`` holds a split written by ``save_to_disk``.

    ``Dataset.load_from_disk`` is the same reader the training code uses, so it
    is the correct probe; the old ``DatasetInfo.from_directory`` helper is not
    public API and could delete a perfectly valid split.
    """
    if not os.path.isdir(path):
        return False
    if not os.path.exists(os.path.join(path, "dataset_info.json")):
        return False
    try:
        Dataset.load_from_disk(path)
        return True
    except Exception as exc:
        print(f"  WARNING: corrupted saved dataset at {path}")
        print(f"  {type(exc).__name__}: {exc}")
        print(f"  Deleting {path} and re-processing...")
        shutil.rmtree(path, ignore_errors=True)
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
    except TypeError as exc:
        if "dataclass" not in str(exc).lower():
            raise
        dataset_name = args[0] if args else kwargs.get("path")
        config_name = args[1] if len(args) > 1 else kwargs.get("name")
        _clear_builder_cache(dataset_name, config_name)
        return load_dataset(*args, **kwargs)


# ------------------------------------------------------------------
# Measured domain shares (cached between runs)
# ------------------------------------------------------------------
def _load_cached_shares():
    """Read the shares measured on a previous run.

    Returns ``(shares, population)`` or ``None``.  The cache is what lets a
    re-run decide whether the existing splits still match the corpus without
    loading and hashing 5.49M documents again.
    """
    if not os.path.exists(DOMAIN_SHARES_FILE):
        return None
    try:
        with open(DOMAIN_SHARES_FILE, "r") as f:
            payload = json.load(f)
        shares = {str(d): float(w)
                  for d, w in (payload.get("shares") or {}).items()}
        if not shares:
            return None
        return shares, payload.get("population")
    except Exception as exc:
        print(f"  WARNING: unreadable {DOMAIN_SHARES_FILE}; re-measuring.")
        print(f"  {type(exc).__name__}: {exc}")
        return None


def _save_cached_shares(shares, population):
    os.makedirs(os.path.dirname(DOMAIN_SHARES_FILE), exist_ok=True)
    with open(DOMAIN_SHARES_FILE, "w") as f:
        json.dump({"population": population, "shares": shares}, f, indent=2)

def _print_shares(shares, population, label):
    print(f"  Domain shares [{label}] over {population:,} rows:")
    for domain in sorted(shares, key=lambda d: -shares[d]):
        share = shares[domain]
        print(f"    {domain:26s} {share:9.6f}  ({100 * share:6.3f}%)")


def _print_achieved_margins(stat_sizes):
    """Half-width actually attained by each split, finite and infinite view."""
    p = stat_sizes["MINORITY_SHARE"]
    conf = stat_sizes["MINORITY_CONFIDENCE"]
    pop = stat_sizes["TOTAL_POPULATION"]
    print("  Achieved relative half-width of the minority (SE) share:")
    for label, key in (("pretrain", "PRETRAIN_SIZE"),
                       ("pretrain_val", "PRETRAIN_VAL_SIZE"),
                       ("finetune", "FINETUNE_SIZE"),
                       ("finetune_val", "FINETUNE_VAL_SIZE")):
        n = stat_sizes[key]
        finite = relative_half_width(n, p, conf, pop)
        infinite = relative_half_width(n, p, conf)
        print(f"    {label:13s} n={n:>9,}  finite={finite:.5f}  infinite={infinite:.5f}")


def _print_stat_sizes(stat_sizes):
    pop = stat_sizes["TOTAL_POPULATION"]
    hw = stat_sizes["MINORITY_RELATIVE_HALF_WIDTH"]
    hw_target = stat_sizes["MINORITY_TARGET_HALF_WIDTH"]
    total_rows = stat_sizes["TOTAL_SAMPLED_ROWS"]
    used = "finite" if stat_sizes["USE_FINITE_CORRECTION"] else "infinite"
    print("  " + "-" * 58)
    print(f"  Domain share source : {stat_sizes['DOMAIN_SHARE_SOURCE']}")
    print(f"  Minority share (SE) : {stat_sizes['MINORITY_SHARE']:.6f}")
    print(f"  Minority half-width : {hw:.5f} (target {hw_target:.5f}, relative SE doc share)")
    print(f"  Cochran n (infinite): {stat_sizes['COCHRAN_INFINITE_N']:,}")
    print(f"  Cochran n (finite)  : {stat_sizes['COCHRAN_FINITE_N']:,} (FPC {stat_sizes['FINITE_POPULATION_FACTOR']:.6f})")
    print(f"  Cochran n used      : {stat_sizes['COCHRAN_TOTAL_N']:,} ({used} population)")
    print(f"  Cochran minority N  : {stat_sizes['COCHRAN_MINORITY_N']:,} (SE quota)")
    print(f"  Pretrain pool size  : {stat_sizes['PRETRAIN_POOL_SIZE']:,}")
    print(f"  Pretrain size       : {stat_sizes['PRETRAIN_SIZE']:,}")
    print(f"  Pretrain val size   : {stat_sizes['PRETRAIN_VAL_SIZE']:,}")
    print(f"  Finetune pool size  : {stat_sizes['FINETUNE_POOL_SIZE']:,}")
    print(f"  Finetune size       : {stat_sizes['FINETUNE_SIZE']:,}")
    print(f"  Finetune val size   : {stat_sizes['FINETUNE_VAL_SIZE']:,}")
    if pop:
        pct = 100 * total_rows / pop
        print(f"  Total sampled rows  : {total_rows:,} ({pct:.2f}% of {pop:,})")
    else:
        print(f"  Total sampled rows  : {total_rows:,}")
    print("  Planned pretrain partition (Hamilton, exact):")
    for domain, n_rows in sorted(stat_sizes["PRETRAIN_DOMAIN_TARGETS"].items(),
                                 key=lambda kv: -kv[1]):
        print(f"    {domain:26s} {n_rows:>9,}")
    _print_achieved_margins(stat_sizes)
    print("  " + "-" * 58)


def _slimpajama_params(max_seq_len, stat_sizes, domain_share_source,
                       sample_ratio=SAMPLE_RATIO):
    """Build parameters must include the weights and ``SAMPLE_RATIO``: changing
    either changes the saved rows, so a stale split has to be rebuilt."""
    return {
        "max_seq_len":         max_seq_len,
        "domain_share_source": domain_share_source,
        "domain_shares":       {d: round(w, 8)
                                for d, w in stat_sizes["DOMAIN_SHARES"].items()},
        "sample_ratio":        round(float(sample_ratio), 8),
        "use_finite_correction": bool(stat_sizes["USE_FINITE_CORRECTION"]),
        "cochran_infinite_n":  stat_sizes["COCHRAN_INFINITE_N"],
        "cochran_finite_n":    stat_sizes["COCHRAN_FINITE_N"],
        "pretrain_pool_size":  stat_sizes["PRETRAIN_POOL_SIZE"],
        "pretrain_seed":       PRETRAIN_SEED,
        "pretrain_size":       subsample_size(stat_sizes["PRETRAIN_SIZE"], sample_ratio),
        "pretrain_val_size":   subsample_size(stat_sizes["PRETRAIN_VAL_SIZE"], sample_ratio),
        "finetune_pool_size":  stat_sizes["FINETUNE_POOL_SIZE"],
        "finetune_seed":       FINETUNE_SEED,
        "finetune_size":       subsample_size(stat_sizes["FINETUNE_SIZE"], sample_ratio),
        "finetune_val_size":   subsample_size(stat_sizes["FINETUNE_VAL_SIZE"], sample_ratio),
    }

# ------------------------------------------------------------------
# Arrow helpers (keep the heavy columns out of Python)
# ------------------------------------------------------------------
def _arrow_column(ds, name):
    try:
        return ds.data.column(name)
    except Exception:
        return None


def _to_numpy(ds, name, dtype=None):
    """Numeric column as a numpy array; zero-copy for Arrow primitives."""
    column = _arrow_column(ds, name)
    if column is not None:
        try:
            if hasattr(column, "combine_chunks"):
                column = column.combine_chunks()
            values = column.to_numpy(zero_copy_only=False)
            return values if dtype is None else values.astype(dtype, copy=False)
        except Exception:
            pass
    return np.asarray(ds[name], dtype=dtype)


def _factorize(values):
    """(int32 codes, labels) for a label column, encoded by Arrow when possible."""
    try:
        encoded = pc.dictionary_encode(pc.fill_null(values, "Unknown"))
        if hasattr(encoded, "combine_chunks"):
            encoded = encoded.combine_chunks()
        codes = encoded.indices.to_numpy(zero_copy_only=False)
        return codes.astype(np.int32, copy=False), encoded.dictionary.to_pylist()
    except Exception:
        codes, labels = pd.factorize(values)
        return np.asarray(codes, dtype=np.int32), [str(x) for x in labels]


def _index_map(codes, labels):
    """{label: sorted positions} from factorisation codes (one stable sort)."""
    codes = np.asarray(codes)
    if codes.size == 0:
        return {}
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=len(labels))
    starts = np.concatenate(([0], np.cumsum(counts)))
    return {str(labels[i]): order[starts[i]:starts[i + 1]]
            for i in range(len(labels)) if counts[i]}


def _codes_counts(codes, labels):
    """Counter of label frequencies for a subset of factorisation codes."""
    codes = np.asarray(codes)
    if codes.size == 0:
        return Counter()
    counts = np.bincount(codes, minlength=len(labels))
    return Counter({str(labels[i]): int(counts[i])
                    for i in np.flatnonzero(counts)})


def _subset_index_map(index_map, drop_mask):
    """Restrict {domain: positions} to the rows where ``drop_mask`` is False."""
    return {d: idx[~drop_mask[idx]] for d, idx in index_map.items()}


# ------------------------------------------------------------------
# Tokenisation (batched + parallel)
# ------------------------------------------------------------------
def _tokenize_batch(batch, max_len):
    tokenizer = _ensure_setup()
    return tokenizer(batch["text"], truncation=True, max_length=max_len,
                     padding=False, return_attention_mask=True)


def _as_examples(encoded):
    for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
        yield {"input_ids": ids, "attention_mask": mask}


def manual_tokenize(text_iter, max_len, batch_size=8192):
    """Kept for backwards compatibility; the builders use _tokenize_dataset."""
    _ensure_setup()
    buffer = []
    for text in text_iter:
        buffer.append(text)
        if len(buffer) >= batch_size:
            yield from _as_examples(_tokenize_batch({"text": buffer}, max_len))
            buffer = []
    if buffer:
        yield from _as_examples(_tokenize_batch({"text": buffer}, max_len))


def _tokenize_dataset(ds, max_len, num_proc=NUM_PROC):
    _ensure_setup()
    return ds.map(
        partial(_tokenize_batch, max_len=max_len),
        batched=True,
        batch_size=TOKENIZE_BATCH,
        writer_batch_size=1000,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        features=LM_FEATURES,
        desc="Tokenizing",
    )


def _total_tokens(ds):
    lengths = pc.list_value_length(ds.with_format("arrow")["attention_mask"])
    return int(pc.sum(lengths).as_py() or 0)


# ------------------------------------------------------------------
# Hashing / deduplication
# ------------------------------------------------------------------
def _text_hash64(text):
    """First 8 bytes of the SHA-256 digest as a signed int64.

    Truncating to 64 bits keeps the dedup index at 44 MB instead of several GB;
    for 5.5M documents the collision probability is ~1e-6.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=True)


def _hash_batch(batch):
    """64-bit SHA-256 digests for one batch, built in a single buffer."""
    digests = b"".join(hashlib.sha256(t.encode("utf-8")).digest()[:8]
                       for t in batch["text"])
    return {"h": np.frombuffer(digests, dtype="<i8").astype(np.int64)}


def _hash_dataset(ds, num_proc=NUM_PROC):
    """One int64 column of SHA-256 digests (44 MB for 5.5M documents).

    ``select_columns`` keeps this a zero-copy view of the cached Arrow table, so
    only the digest column is written - the old pass rewrote the full text
    column just to attach the hash.
    """
    return ds.select_columns(["text"]).map(
        _hash_batch, batched=True, batch_size=HASH_BATCH, num_proc=num_proc,
        remove_columns=["text"], features=HASH_FEATURES,
        desc="Hashing documents",
    )


def _label_batch(batch):
    return {"domain": [(meta or {}).get("redpajama_set_name", "Unknown")
                       for meta in batch["meta"]]}


def _domain_column(ds, num_proc=NUM_PROC):
    """Domain labels as an Arrow column (struct field read, no Python dicts)."""
    try:
        return pc.struct_field(ds.data.column("meta"), "redpajama_set_name")
    except Exception:
        labelled = ds.select_columns(["meta"]).map(
            _label_batch, batched=True, batch_size=8192, num_proc=num_proc,
            remove_columns=["meta"], features=DOMAIN_FEATURES,
            desc="Extracting domain labels",
        )
        return labelled.data.column("domain")


def _deduplicate(ds):
    """(kept row positions, kept hashes) after exact-duplicate removal.

    Returns indices instead of a copy of the dataset: every later split is one
    ``take`` of the original memory-mapped table, so 5.5M documents are never
    rewritten just to drop ~16 duplicate rows.
    """
    hashes = _to_numpy(ds, "h", np.int64)
    keep = np.sort(np.unique(hashes, return_index=True)[1])
    return keep, hashes[keep]

# ------------------------------------------------------------------
# Domain helpers
# ------------------------------------------------------------------
def _extract_minipile_domain(example):
    return {"domain": (example.get("meta") or {}).get("pile_set_name", "Unknown")}


def _domain_indices(domain_array):
    """Return {domain: sorted positions} for any label array."""
    codes, labels = _factorize(domain_array)
    return _index_map(codes, labels)


def _capacities(index_map):
    return {domain: len(idx) for domain, idx in index_map.items()}


# ------------------------------------------------------------------
# 1D Proportional Stratified Sampling
# ------------------------------------------------------------------
def _draw_exact(index_map, allocation, seed):
    """Draw exactly ``allocation[d]`` rows from every stratum.

    ``index_map`` holds absolute row positions, so the result is directly
    usable as a ``Dataset.select`` argument.  Rows are drawn without
    replacement inside each stratum, so the totals are exact.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for domain, n in allocation.items():
        if n <= 0:
            continue
        pool = index_map.get(domain)
        available = 0 if pool is None else len(pool)
        if available < n:
            raise ValueError(
                f"Stratum '{domain}' has {available:,} rows but {n:,} requested"
            )
        parts.append(pool[rng.choice(available, size=n, replace=False)])
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(parts))


def _draw_stratified(index_map, capacities, total, weights, seed):
    allocation = allocate_with_capacity(total, weights, capacities)
    return _draw_exact(index_map, allocation, seed), allocation


def _subsample_indices(indices, ratio, seed):
    """Uniform random subset of a split that has already been designed.

    ``SAMPLE_RATIO`` is a compute-saving copy step, not part of the design: the
    stratified split is drawn in full first, then this many rows are kept with
    equal probability.  A uniform subset of a proportionally allocated split is
    still a simple random sample of the corpus, so the domain mixture is
    preserved in expectation and the four splits stay disjoint.
    """
    n = len(indices)
    m = subsample_size(n, ratio)
    if m >= n:
        return indices
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(n, size=m, replace=False))
    return indices[picked]


def _print_subsample(name, n_design, n_kept, ratio):
    pct = 0.0 if n_design == 0 else 100.0 * n_kept / n_design
    print(f"    {name:26s} {n_design:>9,} -> {n_kept:>9,} ({pct:5.2f}% of design, ratio={ratio})")


def _proportional_stratified_sample(ds, total_n, seed, domain_shares, targets=None):
    """Capacity-aware 1D proportional stratified sample (kept for compatibility)."""
    index_map = _domain_indices(ds["domain"])
    weights = targets if targets is not None else domain_shares
    chosen, _ = _draw_stratified(index_map, _capacities(index_map),
                                 total_n, weights, seed)
    return ds.select(chosen)


def _proportional_stratified_split(ds, val_size, seed, val_weights=None):
    """Stratified train/validation split with an exact per-stratum validation
    size that follows the realised pool mixture.
    """
    index_map = _domain_indices(ds["domain"])
    capacities = _capacities(index_map)
    weights = capacities if val_weights is None else val_weights
    val_alloc = allocate_with_capacity(val_size, weights, capacities)
    val_idx = _draw_exact(index_map, val_alloc, seed)

    val_mask = np.zeros(len(ds), dtype=bool)
    val_mask[val_idx] = True
    return ds.select(np.flatnonzero(~val_mask)), ds.select(val_idx)


def _print_domain_counts(counts, label, expected=None):
    """Print observed rows next to the planned target for every stratum."""
    counts = Counter(counts)
    total = sum(counts.values()) or 1
    keys = set(counts) | set(expected or {})
    print(f"  [{label}] {total:,} rows")
    for domain in sorted(keys, key=lambda d: (-counts.get(d, 0), str(d))):
        target = ""
        if expected is not None and domain in expected:
            target = f"  (target {expected[domain]:,})"
        value = counts.get(domain, 0)
        pct = 100 * value / total
        print(f"    {str(domain):26s} {value:>9,} ({pct:5.3f}%){target}")


def _print_capacity_caps(label, requested, realised):
    """Report strata that could not be filled to their proportional target."""
    capped = {d: (requested[d], realised[d])
              for d in requested if realised[d] < requested[d]}
    if not capped:
        return
    print(f"  [{label}] capacity cap applied (requested -> drawn):")
    for domain in sorted(capped, key=lambda d: realised[d] - requested[d]):
        want, got = capped[domain]
        print(f"    {domain:26s} {want:>9,} -> {got:>9,} ({got - want:+,})")


# ------------------------------------------------------------------
# Statistical verification
# ------------------------------------------------------------------
def _verify_domain_distribution(counts, expected, label):
    """Pearson chi-square goodness of fit against the attainable allocation.

    ``counts`` is the observed per-domain Counter; ``expected`` must be the
    allocation that was actually requested (capacity-aware), not a raw share,
    otherwise strata that cannot be filled always fail the test.  A single
    stratum has nothing to compare, so it is reported as skipped rather than as
    a vacuous "Pass".
    """
    observed = Counter(counts)
    keys = sorted(d for d in expected if expected[d] > 0 or observed.get(d, 0) > 0)

    if len(keys) < 2:
        only = ", ".join(str(k) for k in keys) if keys else "none"
        print(f"  [{label}] Chi-Square skipped: single stratum ({only}); the test needs >= 2 strata.")
        return

    zero_expected = [d for d in keys if expected.get(d, 0) <= 0]
    if zero_expected:
        names = ", ".join(str(d) for d in zero_expected)
        print(f"  [{label}] Chi-Square skipped: expected count is 0 for {names}.")
        return

    obs = np.array([observed.get(d, 0) for d in keys], dtype=np.int64)
    exp = np.array([expected[d] for d in keys], dtype=np.int64)
    if obs.sum() != exp.sum():
        raise ValueError(f"[{label}] observed rows ({obs.sum():,}) != expected ({exp.sum():,})")

    stat, p_value = chisquare(obs, f_exp=exp)
    verdict = "Pass" if p_value > CHI_SQUARE_P_THRESHOLD else "WARNING"
    print(f"  [{label}] Chi-Square: stat={stat:.2f}, p={p_value:.4f} -> {verdict}")
    if verdict == "WARNING":
        print(f"  [{label}] WARNING: p <= {CHI_SQUARE_P_THRESHOLD}, split deviates from plan.")


# ------------------------------------------------------------------
# KL divergence diagnostics
# ------------------------------------------------------------------
def _bincount(texts, vocab_size, max_len=None, batch_size=1024):
    """Token counts for an iterable of texts, one ``bincount`` per batch.

    ``max_len`` truncates exactly like the saved splits, so the reference
    distribution and the split distributions are measured on the same token
    budget.  Passing ``None`` tokenises without truncation.
    """
    tokenizer = _ensure_setup()
    vocab_size = int(vocab_size)
    counts = np.zeros(vocab_size, dtype=np.int64)
    buffer = []

    def flush():
        nonlocal counts
        kwargs = {"truncation": True, "max_length": int(max_len)} if max_len else {}
        encoded = tokenizer(buffer, **kwargs)["input_ids"]
        buffer.clear()
        parts = [np.asarray(ids, dtype=np.int64) for ids in encoded if len(ids)]
        if not parts:
            return
        merged = np.concatenate(parts)
        merged = merged[(merged >= 0) & (merged < vocab_size)]
        if merged.size:
            counts += np.bincount(merged, minlength=vocab_size)

    for text in texts:
        buffer.append(text)
        if len(buffer) >= batch_size:
            flush()
    if buffer:
        flush()
    return counts


def _kl_from_counts(p_counts, q_counts, eps=1e-12):
    """``KL(p || q)`` in nats from two token-count vectors."""
    p = np.asarray(p_counts, dtype=np.float64).ravel()
    q = np.asarray(q_counts, dtype=np.float64).ravel()
    p_total, q_total = p.sum(), q.sum()
    if p_total <= 0 or q_total <= 0:
        return 0.0
    p = p / p_total
    q = np.maximum(q / q_total, eps)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def _kl_noise_floor(p_counts, q_counts):
    """Expected plug-in KL of two samples drawn from the *same* distribution.

    The plug-in estimator is biased upwards by roughly ``(K-1)/(2n)`` per
    sample, which dominates the 1e-3 threshold at real token counts.  Reporting
    the floor next to the estimate separates a real mixture shift from finite
    sampling noise.
    """
    p = np.asarray(p_counts, dtype=np.float64).ravel()
    q = np.asarray(q_counts, dtype=np.float64).ravel()
    n_p, n_q = p.sum(), q.sum()
    if n_p <= 0 or n_q <= 0:
        return 0.0
    support = np.count_nonzero(p + q)
    if support <= 1:
        return 0.0
    return 0.5 * (support - 1.0) * (1.0 / n_p + 1.0 / n_q)


def _verify_kl_divergence(target_counts, reference_counts, label):
    """Compare a saved split's token distribution with the corpus reference."""
    kl = _kl_from_counts(target_counts, reference_counts)
    floor = _kl_noise_floor(target_counts, reference_counts)
    limit = max(KL_DIVERGENCE_THRESHOLD, 2.0 * floor)
    verdict = "Pass" if kl <= limit else "WARNING"
    print(
        f"  [{label}] KL vs corpus: {kl:.6f} nats "
        + f"(noise floor {floor:.6f}, threshold {KL_DIVERGENCE_THRESHOLD}) "
        + f"-> {verdict}"
    )
    if verdict == "WARNING":
        print(f"  [{label}] WARNING: KL exceeds both the threshold and the estimator noise.")
    return {"kl": kl, "noise_floor": floor, "limit": limit, "verdict": verdict}


def _token_counts_from_dataset(ds, vocab_size, max_docs=None):
    """Token counts of an already-tokenised split, via one Arrow scan."""
    if max_docs is not None and len(ds) > int(max_docs):
        ds = ds.select(range(int(max_docs)))
    vocab_size = int(vocab_size)
    try:
        flat = pc.list_flatten(ds.data.column("input_ids"))
        value_counts = pc.value_counts(flat)
        if hasattr(value_counts, "combine_chunks"):
            value_counts = value_counts.combine_chunks()
        values = value_counts.field("values").to_numpy(zero_copy_only=False)
        counts = value_counts.field("counts").to_numpy(zero_copy_only=False)
        out = np.zeros(vocab_size, dtype=np.int64)
        values = values.astype(np.int64, copy=False)
        ok = (values >= 0) & (values < vocab_size)
        out[values[ok]] = counts.astype(np.int64, copy=False)[ok]
        return out
    except Exception as exc:
        print(f"  WARNING: Arrow token count failed ({exc}); using a Python scan.")

    out = np.zeros(vocab_size, dtype=np.int64)
    for batch in ds.iter(batch_size=1024):
        for ids in batch["input_ids"]:
            if not ids:
                continue
            arr = np.asarray(ids, dtype=np.int64)
            arr = arr[(arr >= 0) & (arr < vocab_size)]
            if arr.size:
                out += np.bincount(arr, minlength=vocab_size)
    return out


def _reference_token_counts(ds, keep, max_docs, max_len, vocab_size, seed=PRETRAIN_SEED):
    """Token counts of a bounded random sample of the deduplicated corpus."""
    keep = np.asarray(keep)
    if max_docs is not None and keep.size > int(max_docs):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(keep, size=int(max_docs), replace=False))
    subset = ds.select(keep.tolist()) if keep.size else ds.select([])
    return _bincount(subset["text"], vocab_size, max_len=max_len)


# ------------------------------------------------------------------
# Split construction helpers
# ------------------------------------------------------------------
def _partition_index_map(index_map, selected):
    """Split ``{domain: positions}`` into (selected, rest) with one mask pass."""
    selected = np.asarray(selected)
    chosen, rest = {}, {}
    for domain, positions in index_map.items():
        mask = selected[positions]
        chosen[domain] = positions[mask]
        rest[domain] = positions[~mask]
    return chosen, rest


def _splits_ready(names):
    return all(_dataset_saved(PATHS[name]) for name in names)


def _tokenize_and_save(ds, positions, path, max_len, num_proc=NUM_PROC):
    """One ``select`` + one parallel ``map`` + ``save_to_disk``.

    ``positions`` are absolute rows into the original memory-mapped table, so no
    intermediate dataset is materialised.
    """
    subset = ds.select(np.asarray(positions).tolist())
    tokenized = _tokenize_dataset(subset, max_len, num_proc=num_proc)
    tokenized.save_to_disk(path)
    return tokenized


def _initial_share_state():
    """(shares, population, source) available without touching the corpus."""
    if DOMAIN_SHARE_SOURCE == "measured":
        cached = _load_cached_shares()
        if cached is not None:
            shares, population = cached
            try:
                population = int(population) if population else None
            except (TypeError, ValueError):
                population = None
            return shares, population, "measured"
    return dict(PAPER_DOMAIN_SHARES), None, "paper (Table 4)"


SLIMPAJAMA_SPLITS = ("slimpajama_train", "slimpajama_val",
                     "slimpajama_finetune", "slimpajama_finetune_val")


# ------------------------------------------------------------------
# SlimPajama builder
# ------------------------------------------------------------------
def build_slimpajama(max_seq_len=MAX_SEQ_LEN,
                     minority_confidence=MINORITY_CONFIDENCE,
                     minority_margin_error=MINORITY_MARGIN_ERROR,
                     force_rebuild=False,
                     use_finite_correction=None,
                     sample_ratio=SAMPLE_RATIO,
                     num_proc=NUM_PROC):
    """Build the four disjoint SlimPajama splits.

    Order of operations: measure -> design -> draw -> subsample -> tokenise.
    ``SAMPLE_RATIO`` is applied after the exact stratified draw, so the design
    statistics (and therefore the cache key) stay independent of it while the
    saved rows shrink uniformly.
    """
    print("=" * 72)
    print("SlimPajama: 1D proportional stratified sample")
    print("=" * 72)

    # ---- 1. Reuse check without touching the corpus -------------------
    shares, population, source = _initial_share_state()
    preview = get_stratified_sizes(
        minority_confidence, minority_margin_error, shares, population,
        source, use_finite_correction=use_finite_correction,
    )
    preview_params = _slimpajama_params(max_seq_len, preview, source, sample_ratio)
    ready = False if force_rebuild else _splits_ready(SLIMPAJAMA_SPLITS)
    if ready and _params_match(_load_meta().get("slimpajama"), preview_params):
        print(f"  Existing splits match ({source}); nothing to do.")
        return preview_params

    # ---- 2. Load, hash, deduplicate, label ----------------------------
    print("  Loading DKYoon/SlimPajama-6B ...")
    ds = _safe_load_dataset("DKYoon/SlimPajama-6B", split="train")
    raw_rows = len(ds)
    print(f"  Raw rows: {raw_rows:,}")

    hashed = _hash_dataset(ds, num_proc=num_proc)
    keep, kept_hashes = _deduplicate(hashed)
    del hashed
    np.save(DEDUP_HASH_FILE, kept_hashes)
    print(f"  Deduplicated rows: {keep.size:,} ({raw_rows - keep.size:,} duplicates)")

    codes_all, labels = _factorize(_domain_column(ds, num_proc=num_proc))
    codes = codes_all[keep]
    del codes_all
    population = int(keep.size)

    counts = _codes_counts(codes, labels)
    if DOMAIN_SHARE_SOURCE == "measured":
        shares = domain_shares_from_counts(counts)
        source = "measured"
        _save_cached_shares(shares, population)
    else:
        shares = dict(PAPER_DOMAIN_SHARES)
        source = "paper (Table 4)"
    _print_shares(shares, population, source)

    stat_sizes = get_stratified_sizes(
        minority_confidence, minority_margin_error, shares, population,
        source, use_finite_correction=use_finite_correction,
    )
    _print_stat_sizes(stat_sizes)
    params = _slimpajama_params(max_seq_len, stat_sizes, source, sample_ratio)

    if ready and _params_match(_load_meta().get("slimpajama"), params):
        print("  Existing splits already match the measured corpus; nothing to do.")
        return params

    # ---- 3. Design the exact split sizes (dedup row space) ------------
    index_map = _index_map(codes, labels)
    capacities = _capacities(index_map)

    pool_targets = {
        d: stat_sizes["PRETRAIN_DOMAIN_TARGETS"].get(d, 0)
           + stat_sizes["PRETRAIN_VAL_DOMAIN_TARGETS"].get(d, 0)
        for d in shares
    }
    pool_alloc = allocate_with_capacity(stat_sizes["PRETRAIN_POOL_SIZE"],
                                        pool_targets, capacities)
    _print_capacity_caps("pretrain pool", pool_targets, pool_alloc)

    pool_idx = _draw_exact(index_map, pool_alloc, PRETRAIN_SEED)
    pool_mask = np.zeros(population, dtype=bool)
    pool_mask[pool_idx] = True
    pool_map, remaining_map = _partition_index_map(index_map, pool_mask)

    val_alloc = allocate_with_capacity(stat_sizes["PRETRAIN_VAL_SIZE"],
                                       pool_alloc, pool_alloc)
    val_idx = _draw_exact(pool_map, val_alloc, PRETRAIN_SEED + 1)
    val_mask = np.zeros(population, dtype=bool)
    val_mask[val_idx] = True
    pretrain_design = pool_idx[~val_mask[pool_idx]]

    _verify_domain_distribution(_codes_counts(codes[pretrain_design], labels),
                                stat_sizes["PRETRAIN_DOMAIN_TARGETS"], "pretrain")
    _verify_domain_distribution(_codes_counts(codes[val_idx], labels),
                                stat_sizes["PRETRAIN_VAL_DOMAIN_TARGETS"], "pretrain_val")
    _print_domain_counts(_codes_counts(codes[pretrain_design], labels),
                         "pretrain", stat_sizes["PRETRAIN_DOMAIN_TARGETS"])

    remaining_caps = {d: capacities.get(d, 0) - pool_alloc.get(d, 0) for d in shares}
    ft_targets = {
        d: stat_sizes["FINETUNE_DOMAIN_TARGETS"].get(d, 0)
           + stat_sizes["FINETUNE_VAL_DOMAIN_TARGETS"].get(d, 0)
        for d in shares
    }
    ft_pool_alloc = allocate_with_capacity(stat_sizes["FINETUNE_POOL_SIZE"],
                                           ft_targets, remaining_caps)
    _print_capacity_caps("finetune pool", ft_targets, ft_pool_alloc)

    ft_pool_idx = _draw_exact(remaining_map, ft_pool_alloc, FINETUNE_SEED)
    ft_pool_mask = np.zeros(population, dtype=bool)
    ft_pool_mask[ft_pool_idx] = True
    ft_pool_map, _ = _partition_index_map(remaining_map, ft_pool_mask)

    ft_val_alloc = allocate_with_capacity(stat_sizes["FINETUNE_VAL_SIZE"],
                                          ft_pool_alloc, ft_pool_alloc)
    ft_val_idx = _draw_exact(ft_pool_map, ft_val_alloc, FINETUNE_SEED + 1)
    ft_val_mask = np.zeros(population, dtype=bool)
    ft_val_mask[ft_val_idx] = True
    finetune_design = ft_pool_idx[~ft_val_mask[ft_pool_idx]]

    _verify_domain_distribution(_codes_counts(codes[finetune_design], labels),
                                stat_sizes["FINETUNE_DOMAIN_TARGETS"], "finetune")
    _verify_domain_distribution(_codes_counts(codes[ft_val_idx], labels),
                                stat_sizes["FINETUNE_VAL_DOMAIN_TARGETS"], "finetune_val")

    # ---- 4. SAMPLE_RATIO: uniform post-design subsample ----------------
    print("  Post-design SAMPLE_RATIO (uniform, seeds %d-%d):"
          % (SAMPLE_SEED, SAMPLE_SEED + 3))
    pretrain_idx = _subsample_indices(pretrain_design, sample_ratio, SAMPLE_SEED)
    pretrain_val_idx = _subsample_indices(val_idx, sample_ratio, SAMPLE_SEED + 1)
    finetune_idx = _subsample_indices(finetune_design, sample_ratio, SAMPLE_SEED + 2)
    finetune_val_idx = _subsample_indices(ft_val_idx, sample_ratio, SAMPLE_SEED + 3)
    for label, designed, kept in (
        ("pretrain", pretrain_design, pretrain_idx),
        ("pretrain_val", val_idx, pretrain_val_idx),
        ("finetune", finetune_design, finetune_idx),
        ("finetune_val", ft_val_idx, finetune_val_idx),
    ):
        _print_subsample(label, len(designed), len(kept), sample_ratio)

    # ---- 5. Tokenise + save (one select/map per split) ----------------
    outputs = (
        ("pretrain", pretrain_idx, PATHS["slimpajama_train"]),
        ("pretrain_val", pretrain_val_idx, PATHS["slimpajama_val"]),
        ("finetune", finetune_idx, PATHS["slimpajama_finetune"]),
        ("finetune_val", finetune_val_idx, PATHS["slimpajama_finetune_val"]),
    )
    token_counts = {}
    for label, idx, path in outputs:
        print(f"  Tokenising {label}: {len(idx):,} rows -> {path}")
        tokenized = _tokenize_and_save(ds, keep[idx], path, max_seq_len,
                                       num_proc=num_proc)
        token_counts[label] = _token_counts_from_dataset(
            tokenized, VOCAB_SIZE, KL_TARGET_MAX_DOCS)
        print(f"    {int(token_counts[label].sum()):,} tokens")
        del tokenized

    # ---- 6. KL diagnostics against the deduplicated corpus ------------
    print("  KL diagnostics (saved split vs corpus reference):")
    reference_counts = _reference_token_counts(
        ds, keep, KL_REFERENCE_MAX_DOCS, max_seq_len, VOCAB_SIZE)
    for label in ("pretrain", "finetune"):
        _verify_kl_divergence(token_counts[label], reference_counts, label)
    del reference_counts

    _record_built("slimpajama", params)
    print(
        f"  SlimPajama done: pretrain {len(pretrain_idx):,}, "
        + f"val {len(pretrain_val_idx):,}, finetune {len(finetune_idx):,}, "
        + f"ft_val {len(finetune_val_idx):,}"
    )
    return params


# ------------------------------------------------------------------
# MiniPile builder
# ------------------------------------------------------------------
def _minipile_draw_size(population, ratio, margin_of_error=MINORITY_MARGIN_ERROR,
                        confidence=MINORITY_CONFIDENCE):
    """Rows to keep from the **clean** MiniPile pool.

    ``ratio >= 1`` keeps the whole published split.  Below 1 it is
    ``min(Cochran n, ratio * N)``: the Cochran size for ``p = 0.5`` (MiniPile
    has no domain labels) capped by the smoke-test ratio.  The size is computed
    from the decontaminated population, so the final split is exact.

    Deliberately no finite-population correction here: with ``N ~ 10,000`` the
    correction would shrink 9,604 rows to roughly 4,900 and throw away half of a
    benchmark that is cheap to evaluate.
    """
    population = int(population)
    if population <= 0:
        raise ValueError("MiniPile population must be positive")
    ratio = float(ratio)
    if not (ratio > 0):
        raise ValueError(f"bench sample ratio must be positive, got {ratio}")

    _, info = benchmark_sample_size(
        population, margin_of_error=margin_of_error, confidence=confidence)
    if ratio >= 1.0:
        return population, dict(info, ratio=ratio, draw_n=population,
                                limit="full split")

    cap = subsample_size(population, ratio)
    n, info = benchmark_sample_size(
        population, margin_of_error=margin_of_error, confidence=confidence,
        cap=cap)
    return n, dict(info, ratio=ratio, draw_n=n)


def build_minipile(max_seq_len=MAX_SEQ_LEN, force_rebuild=False,
                   bench_sample_ratio=BENCH_SAMPLE_RATIO,
                   minority_margin_error=MINORITY_MARGIN_ERROR,
                   minority_confidence=MINORITY_CONFIDENCE,
                   num_proc=NUM_PROC):
    """Decontaminate MiniPile first, then size and draw from the clean pool."""
    print("=" * 72)
    print("MiniPile: zero-shot benchmark")
    print("=" * 72)

    path = PATHS["minipile"]
    print("  Loading JeanKaddour/minipile ...")
    ds = _safe_load_dataset("JeanKaddour/minipile", split="test")
    population = len(ds)
    clean_idx = np.arange(population, dtype=np.int64)

    if os.path.exists(DEDUP_HASH_FILE):
        known = np.load(DEDUP_HASH_FILE)
        doc_hashes = _to_numpy(_hash_dataset(ds, num_proc=num_proc), "h", np.int64)
        contaminated = np.isin(doc_hashes, known)
        clean_idx = np.flatnonzero(~contaminated).astype(np.int64, copy=False)
        print(
            f"  Decontamination: {int(contaminated.sum()):,} of {population:,} "
            + f"rows match SlimPajama; {clean_idx.size:,} clean rows remain."
        )
    else:
        print("  WARNING: SlimPajama hash index missing; skipping decontamination.")

    n, info = _minipile_draw_size(clean_idx.size, bench_sample_ratio,
                                  minority_margin_error, minority_confidence)
    print(f"  Clean population {clean_idx.size:,} -> draw {n:,} rows "
          + f"(limit={info['limit']}, Cochran {info['cochran_n']:,})")

    params = {
        "max_seq_len": int(max_seq_len),
        "bench_sample_ratio": round(float(bench_sample_ratio), 8),
        "population": int(population),
        "decontaminated": int(clean_idx.size),
        "benchmark_size": int(n),
        "seed": PRETRAIN_SEED,
    }
    if not force_rebuild and _dataset_saved(path) \
            and _params_match(_load_meta().get("minipile"), params):
        print("  Existing MiniPile split matches; nothing to do.")
        return params

    draw = clean_idx if n >= clean_idx.size else _draw_exact(
        {"minipile": clean_idx}, {"minipile": n}, PRETRAIN_SEED)
    subset = ds.select(np.asarray(draw).tolist())
    if "meta" in subset.column_names:
        subset = subset.map(_extract_minipile_domain, desc="MiniPile domains")
        domain_counts = Counter(subset["domain"])
    else:
        domain_counts = Counter({"Unknown": len(subset)})
    _print_domain_counts(domain_counts, "minipile")
    _verify_domain_distribution(domain_counts, domain_counts, "minipile")

    print(f"  Tokenising MiniPile: {len(subset):,} rows -> {path}")
    tokenized = _tokenize_dataset(subset, max_seq_len, num_proc=num_proc)
    tokenized.save_to_disk(path)
    print(f"    {_total_tokens(tokenized):,} tokens")

    _record_built("minipile", params)
    print("  MiniPile done.")
    return params


# ------------------------------------------------------------------
# Benchmark evaluation
# ------------------------------------------------------------------
def evaluate_zero_shot_with_bootstrap(dataset_path=PATHS["minipile"],
                                      max_seq_len=MAX_SEQ_LEN,
                                      n_bootstrap=BOOTSTRAP_ITERATIONS,
                                      seed=PRETRAIN_SEED,
                                      tokenizer=None):
    """Bootstrap confidence interval for the mean length of a saved split.

    Works on an already-tokenised split (``attention_mask`` / ``input_ids``) and
    falls back to tokenising ``text`` in batches.  Returns the point estimate and
    a 95% percentile interval over ``n_bootstrap`` resamples.
    """
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"no saved dataset at {dataset_path}")
    ds = Dataset.load_from_disk(dataset_path)
    if len(ds) == 0:
        raise ValueError(f"{dataset_path} is empty")

    if "attention_mask" in ds.column_names:
        lengths = np.fromiter((len(x) for x in ds["attention_mask"]),
                              dtype=np.int64, count=len(ds))
    elif "input_ids" in ds.column_names:
        lengths = np.fromiter((len(x) for x in ds["input_ids"]),
                              dtype=np.int64, count=len(ds))
    else:
        tokenizer = tokenizer or _ensure_setup()
        lengths = np.empty(len(ds), dtype=np.int64)
        for start in range(0, len(ds), 1024):
            batch = ds[start:start + 1024]["text"]
            encoded = tokenizer(batch, truncation=True,
                                max_length=max_seq_len)["input_ids"]
            lengths[start:start + len(encoded)] = [len(ids) for ids in encoded]

    rng = np.random.default_rng(seed)
    n = lengths.size
    means = np.empty(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        means[i] = lengths[rng.integers(0, n, size=n)].mean()

    result = {
        "n_docs": int(n),
        "mean_tokens": float(lengths.mean()),
        "median_tokens": float(np.median(lengths)),
        "std_tokens": float(lengths.std()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n_bootstrap": int(n_bootstrap),
    }
    print(
        f"  {os.path.basename(dataset_path)}: {n:,} docs, "
        + f"mean {result['mean_tokens']:.1f} tokens "
        + f"[{result['ci_low']:.1f}, {result['ci_high']:.1f}] "
        + f"({result['n_bootstrap']:,} bootstrap resamples)"
    )
    return result


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def run_preprocessing(max_seq_len=MAX_SEQ_LEN,
                      minority_confidence=MINORITY_CONFIDENCE,
                      minority_margin_error=MINORITY_MARGIN_ERROR,
                      build_slimpajama_data=True,
                      build_minipile_data=False,
                      force_rebuild=False,
                      finite_population_correction=None,
                      slimpajama_sample_ratio=SAMPLE_RATIO,
                      bench_sample_ratio=BENCH_SAMPLE_RATIO):
    """Build the requested splits with the paper's sampling design.

    ``finite_population_correction=None`` follows
    ``config.USE_FINITE_POPULATION_CORRECTION``; pass ``False`` to reproduce the
    published infinite-population sizes.
    """
    print("=" * 72)
    print("Preprocessing pipeline")
    print(f"  max_seq_len={max_seq_len} | confidence={minority_confidence} | "
          + f"relative margin={minority_margin_error}")
    print("=" * 72)

    if finite_population_correction is None:
        finite_population_correction = bool(USE_FINITE_POPULATION_CORRECTION)

    _ensure_setup()
    results = {}

    if build_slimpajama_data:
        results["slimpajama"] = build_slimpajama(
            max_seq_len=max_seq_len,
            minority_confidence=minority_confidence,
            minority_margin_error=minority_margin_error,
            force_rebuild=force_rebuild,
            use_finite_correction=finite_population_correction,
            sample_ratio=slimpajama_sample_ratio,
        )

    if build_minipile_data:
        results["minipile"] = build_minipile(
            max_seq_len=max_seq_len,
            force_rebuild=force_rebuild,
            bench_sample_ratio=bench_sample_ratio,
            minority_margin_error=minority_margin_error,
            minority_confidence=minority_confidence,
        )

    print("=" * 72)
    print("Preprocessing complete.")
    return results

