"""Data-driven 1D proportional stratified sampling for SlimPajama.

The stratification weights are **measured from the corpus that is actually
downloaded** instead of being hard-coded: :func:`domain_shares_from_counts`
turns the per-domain document counts of the loaded dataset into normalised row
proportions, and :func:`get_stratified_sizes` uses those proportions both as the
Cochran minority share and as the allocation weights.

``PAPER_DOMAIN_SHARES`` still carries the published SlimPajama Table-4 shares.
It is a reference / fallback only, because Table 4 counts *tokens* whereas the
counts used here are *documents*: a Book document is ~140k tokens and a C4
document ~1k, so the two sets of proportions are very different.

Mixture note: proportional document allocation gives every document the same
inclusion probability (``n / N``), so the result is a simple random sample of
the corpus in distribution and its token / byte mixture is unbiased.  That is
why ``measured`` document shares are the correct weights even though Table 4
reports token shares; the two only differ because document lengths differ by
domain.

Sample size note: ``get_stratified_sizes`` reports both the infinite-population
Cochran size ``n0`` and the finite-population size ``n0 / (1 + (n0 - 1) / N)``,
and uses the latter by default.  Rows are drawn without replacement from a
finite corpus, so the correction is what actually delivers the requested
relative margin; ``use_finite_correction=False`` reproduces the published
infinite-population figures.
"""
import math
from collections import Counter
from fractions import Fraction
from functools import lru_cache

import scipy.stats as st

# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------
MINORITY_DOMAIN = "RedPajamaStackExchange"

# SlimPajama Table 4 (token shares).  Reference / fallback only - the pipeline
# measures the document shares of the downloaded corpus at build time.
PAPER_DOMAIN_SHARES = {
    "RedPajamaCommonCrawl":   0.522,
    "RedPajamaC4":            0.267,
    "RedPajamaGithub":        0.052,
    "RedPajamaArXiv":         0.046,
    "RedPajamaBook":          0.042,
    "RedPajamaWikipedia":     0.038,
    "RedPajamaStackExchange": 0.033,
}

# Each split keeps 100% of its calculated target; the validation rows are added
# on top of it, so a pool of n_target / (1 - 0.10) is drawn and 10% of that pool
# becomes the validation split.
VALIDATION_FRACTION = Fraction(1, 10)

# Fine-tuning rows are an exact 10% transfer ratio of the pre-training target
# (Hernandez et al., 2021, Scaling Laws for Transfer).
FINETUNE_FRACTION = Fraction(1, 10)

DEFAULT_POPULATION = 5_490_000

# Cochran's finite-population correction
#   n = n0 / (1 + (n0 - 1) / N)
# is applied by default: the parent split has 5.49M rows and the sample is drawn
# WITHOUT replacement, so the exact finite-population size is the correct one.
# Set ``config.USE_FINITE_POPULATION_CORRECTION = False`` (or pass
# ``use_finite_correction=False``) to reproduce the published infinite-population
# figures (~182k rows instead of ~176k).
USE_FINITE_POPULATION_CORRECTION = True


@lru_cache(maxsize=None)
def z_value(confidence=0.95):
    """Two-decimal normal quantile (1.96 for 95%, 2.58 for 99%), cached."""
    return round(float(st.norm.ppf(1 - (1 - confidence) / 2)), 2)


# ---------------------------------------------------------------------------
# Dynamic domain shares (measured from the downloaded dataset)
# ---------------------------------------------------------------------------
def domain_shares_from_counts(counts, domains=None, drop_empty=True):
    """Turn per-stratum document counts into normalised stratification weights.

    Parameters
    ----------
    counts:
        Mapping ``domain -> number of documents`` for the corpus that was
        actually loaded (e.g. ``{d: len(idx) for d, idx in index_map.items()}``
        or a ``Counter`` over the ``domain`` column).
    domains:
        Optional ordered list of domains to keep.  Defaults to every key in
        ``counts`` (sorted, for a deterministic order).
    drop_empty:
        Drop strata with zero documents so they are not given a weight.

    Returns
    -------
    dict
        ``{domain: N_d / N}`` with the weights summing to 1.  These are exactly
        the proportions proportional stratified sampling should reproduce:
        stratum ``d`` receives ``n * N_d / N`` rows.

    Notes
    -----
    Row counts are used because they are cheap and exact: the loaded dataset
    already knows how many rows belong to each domain, no extra pass over 5.49M
    documents is needed.  The consequence is that the resulting mixture is the
    *document* mixture of the corpus, not the token mixture of Table 4.
    """
    keys = list(domains) if domains is not None else sorted(counts)
    total = sum(int(counts.get(d, 0)) for d in keys)
    if total <= 0:
        raise ValueError("domain_shares_from_counts: the corpus has no rows")

    shares = {}
    for domain in keys:
        n_rows = int(counts.get(domain, 0))
        if n_rows <= 0 and drop_empty:
            continue
        shares[str(domain)] = n_rows / total
    return shares


def measure_domain_shares(domain_iterable, domains=None):
    """Count any iterable/column of domain labels and return its row shares."""
    counts = Counter(str(d) for d in domain_iterable)
    return domain_shares_from_counts(counts, domains)


# ---------------------------------------------------------------------------
# Statistical Sample Size Calculation (Cochran's Formula)
# ---------------------------------------------------------------------------
def cochran_sample_size(population_size=None, p=0.5, margin_of_error=0.01,
                        confidence=0.95, finite_population_correction=False):
    """Total sample size needed to estimate a proportion ``p``.

    ``margin_of_error`` is a RELATIVE margin (a fraction of ``p``), matching the
    design statement "+-2% relative error on the stack-exchange stratum at 95%
    confidence".  For an absolute half-width ``e`` the relative one is
    ``e / p``, hence

        n0 = z^2 * p * (1 - p) / (margin_of_error * p)^2
           = z^2 * (1 - p) / (margin_of_error^2 * p)

    ``finite_population_correction`` switches the result from the
    infinite-population size ``n0`` to Cochran's finite-population size
    ``n = n0 / (1 + (n0 - 1) / N)``.  :func:`get_stratified_sizes` turns it on
    by default, because the parent split is finite and rows are drawn without
    replacement; pass ``use_finite_correction=False`` there to reproduce the
    published infinite-population figures.
    """
    z = z_value(confidence)
    n0 = (z ** 2) * (1.0 - p) / (margin_of_error ** 2 * p)
    if finite_population_correction:
        if not population_size:
            raise ValueError("finite_population_correction needs population_size")
        n0 = n0 / (1.0 + (n0 - 1.0) / population_size)
    return int(round(n0))


def finite_population_size(n0, population_size):
    """Cochran's finite-population correction, rounded and clipped to ``N``.

    The infinite-population requirement ``n0`` assumes sampling with
    replacement.  Drawing ``n`` rows without replacement from ``N`` shrinks the
    variance of a proportion by ``(1 - n/N)``, so the size that achieves the
    same margin is

        n = n0 / (1 + (n0 - 1) / N)

    For the SlimPajama corpus (N = 5,488,984) and the 2% relative margin on the
    StackExchange share this is 176,274 rows instead of 182,123.
    """
    n0 = int(n0)
    N = int(population_size or 0)
    if N <= 0:
        return n0
    return max(1, min(N, int(round(n0 / (1.0 + (n0 - 1.0) / N)))))


def _finite_correction_default():
    """Read ``config.USE_FINITE_POPULATION_CORRECTION`` at call time.

    ``config`` imports this module, so the import has to be lazy; resolving the
    flag when the function is called also means a notebook that flips it after
    import takes effect on the next build.
    """
    try:
        from .config import USE_FINITE_POPULATION_CORRECTION as flag
        return bool(flag)
    except Exception:
        return bool(USE_FINITE_POPULATION_CORRECTION)


def relative_half_width(n, p=0.5, confidence=0.95, population_size=None):
    """Relative half-width of a proportion estimated from ``n`` rows.

    Inverse of :func:`cochran_sample_size`; used to report what a split or
    benchmark actually achieves.  Pass ``population_size`` to include the
    finite-population factor ``sqrt(1 - n/N)``, i.e. the margin that is actually
    attained when ``n`` rows are drawn without replacement from ``N``.
    """
    if n <= 0:
        return float("inf")
    half_width = z_value(confidence) * math.sqrt((1.0 - p) / (n * p))
    if population_size:
        half_width *= math.sqrt(max(0.0, 1.0 - n / float(population_size)))
    return half_width


def subsample_size(n, ratio):
    """Rows kept when a finished split of ``n`` rows is subsampled at ``ratio``.

    Used by the post-design ``SAMPLE_RATIO`` step: the stratified split is drawn
    in full first, then this many rows are kept uniformly at random.  ``ratio``
    of 1.0 (or more) keeps the split unchanged.
    """
    n = int(n)
    ratio = float(ratio)
    if not (ratio > 0):
        raise ValueError(f"sample ratio must be positive, got {ratio}")
    if n <= 0:
        return 0
    if ratio >= 1.0:
        return n
    return max(1, min(n, int(math.floor(n * ratio + 0.5))))


def benchmark_sample_size(population_size, margin_of_error=0.02, confidence=0.95,
                          p=0.5, cap=None):
    """Cochran size for a benchmark split that has no domain labels.

    MiniPile rows are ``{"text": ..., "length": ...}`` only - there is no
    category column, so there is no minority stratum to anchor the design on and
    no sensible way to stratify.  The conservative substitute is the
    maximum-variance proportion ``p = 0.5``, which needs no labels and bounds
    every other proportion:

        n0 = z^2 * (1 - p) / (margin^2 * p) = 1.96^2 / 0.02^2 = 9,604

    Note that at ``p = 0.5`` a 2% *relative* margin is also a +-1 percentage
    point *absolute* margin, so 9,604 is the textbook "+-1 pp at 95%" size.

    ``cap`` is an optional upper bound (``BENCH_SAMPLE_RATIO * population``) so
    the ratio keeps working as a quick smoke-test switch.

    Returns ``(n, details)``; ``details["limit"]`` names the binding constraint
    (``"cochran"``, ``"bench_sample_ratio"`` or ``"population"``).
    """
    population_size = int(population_size)
    if population_size <= 0:
        raise ValueError("benchmark_sample_size needs a positive population")

    n0 = cochran_sample_size(population_size, p=p,
                             margin_of_error=margin_of_error,
                             confidence=confidence)
    if n0 >= population_size:
        n, limit = population_size, "population"
    else:
        n, limit = n0, "cochran"
    if cap is not None and int(cap) < n:
        n, limit = int(cap), "bench_sample_ratio"
    n = max(1, n)

    return n, {
        "cochran_n":       n0,
        "population":      population_size,
        "p":               p,
        "margin_of_error": margin_of_error,
        "confidence":      confidence,
        "cap":             None if cap is None else int(cap),
        "limit":           limit,
    }


# ---------------------------------------------------------------------------
# Integer allocation helpers
# ---------------------------------------------------------------------------
def pool_size_for_target(n_target, val_fraction=VALIDATION_FRACTION):
    """Smallest pool whose ``1 - val_fraction`` share still equals ``n_target``.

    n_pool = ceil(n_target / (1 - val_fraction)), evaluated with exact rational
    arithmetic: 28,143 / 0.90 is 31,270.000000000004 in binary floating point,
    which a plain ``math.ceil`` would round up to 31,271 and silently change the
    validation size to 3,128.
    """
    return math.ceil(Fraction(n_target) / (1 - val_fraction))


def largest_remainder(total, weights):
    """Hamilton apportionment: split ``total`` into integers proportional to
    ``weights`` so that the parts add up to exactly ``total``.
    """
    active = {d: w for d, w in weights.items() if w > 0}
    weight_sum = float(sum(active.values()))
    if not active or weight_sum <= 0:
        raise ValueError("largest_remainder needs at least one positive weight")

    raw = {d: total * w / weight_sum for d, w in active.items()}
    alloc = {d: int(math.floor(v)) for d, v in raw.items()}

    remainder = total - sum(alloc.values())
    if remainder:
        order = sorted(raw, key=lambda d: (raw[d] - alloc[d], raw[d]), reverse=True)
        for d in order[:remainder]:
            alloc[d] += 1

    return {d: alloc.get(d, 0) for d in weights}


def allocate_with_capacity(total, weights, capacities):
    """Capacity-aware Hamilton allocation.

    Strata are filled proportionally to ``weights`` but never beyond
    ``capacities[d]``.  When a stratum is exhausted the deficit is
    redistributed proportionally over the remaining strata, so the total is
    always exact and no stratum is over-drawn.
    """
    alloc = {d: 0 for d in weights}
    active = {d: w for d, w in weights.items()
              if w > 0 and capacities.get(d, 0) > 0}
    remaining = total

    while remaining > 0 and active:
        proposal = largest_remainder(remaining, active)
        capped = [d for d in active
                  if proposal[d] > capacities[d] - alloc[d]]

        if not capped:
            for d, value in proposal.items():
                alloc[d] += value
            remaining = 0
            break

        for d in capped:
            remaining -= capacities[d] - alloc[d]
            alloc[d] = capacities[d]
            del active[d]

    if remaining > 0:
        available = sum(capacities.values())
        raise ValueError(f"Cannot allocate {total:,} rows, only {available:,} available")
    return alloc


# ---------------------------------------------------------------------------
# Public sizing API
# ---------------------------------------------------------------------------
def get_stratified_sizes(minority_confidence=0.95, minority_margin_error=0.02,
                         domain_shares=None, population_size=None,
                         share_source="measured", use_finite_correction=None):
    """Compute every split size and per-domain target.

    Parameters
    ----------
    domain_shares:
        Weights measured from the downloaded corpus (see
        :func:`domain_shares_from_counts`).  When omitted the published Table-4
        shares are used, which is only meant for standalone sizing queries.
    population_size:
        Number of rows in the parent split.  It is the ``N`` of the
        finite-population correction and is also used for reporting.
    share_source:
        Human-readable label for the origin of the weights.
    use_finite_correction:
        ``None`` (default) follows ``config.USE_FINITE_POPULATION_CORRECTION``;
        ``True`` applies ``n = n0 / (1 + (n0 - 1) / N)``; ``False`` keeps the
        infinite-population ``n0``.

    The pre-training target is the Cochran sample size for the minority
    (StackExchange) share, evaluated at the requested confidence and RELATIVE
    margin; the pool, validation and fine-tuning splits follow from it.
    """
    if domain_shares is None:
        shares = dict(PAPER_DOMAIN_SHARES)
        share_source = "paper (Table 4 fallback)"
    else:
        shares = {str(d): float(w) for d, w in domain_shares.items() if w > 0}
        if not shares:
            raise ValueError("domain_shares is empty")

    weight_sum = sum(shares.values())
    if weight_sum <= 0:
        raise ValueError("domain_shares must contain a positive total weight")
    if abs(weight_sum - 1.0) > 1e-9:
        shares = {d: w / weight_sum for d, w in shares.items()}

    minority_share = shares.get(MINORITY_DOMAIN, 0.0)
    if minority_share <= 0:
        raise ValueError(f"minority stratum {MINORITY_DOMAIN!r} missing from {sorted(shares)}")

    if population_size is None:
        population_size = _population_from_metadata()
    population_size = int(population_size)

    if use_finite_correction is None:
        use_finite_correction = _finite_correction_default()
    use_finite_correction = bool(use_finite_correction)

    # Infinite-population requirement, then the finite-population one.
    cochran_infinite_n = cochran_sample_size(
        population_size, p=minority_share,
        margin_of_error=minority_margin_error,
        confidence=minority_confidence,
    )
    cochran_finite_n = finite_population_size(cochran_infinite_n, population_size)
    pretrain_size = cochran_finite_n if use_finite_correction else cochran_infinite_n

    pretrain_pool_size = pool_size_for_target(pretrain_size)
    pretrain_val_size = pretrain_pool_size - pretrain_size

    finetune_size = int(round(Fraction(pretrain_size) * FINETUNE_FRACTION))
    finetune_pool_size = pool_size_for_target(finetune_size)
    finetune_val_size = finetune_pool_size - finetune_size

    # Per-domain targets: exact integer partitions, no lost/gained rows
    pretrain_domain_targets = largest_remainder(pretrain_size, shares)
    pretrain_val_domain_targets = largest_remainder(pretrain_val_size, shares)
    finetune_domain_targets = largest_remainder(finetune_size, shares)
    finetune_val_domain_targets = largest_remainder(finetune_val_size, shares)

    half_width = relative_half_width(
        pretrain_size, minority_share, minority_confidence,
        population_size if use_finite_correction else None,
    )

    return {
        "TOTAL_POPULATION": population_size,
        "DOMAIN_SHARES": shares,
        "DOMAIN_SHARE_SOURCE": share_source,
        "MINORITY_DOMAIN": MINORITY_DOMAIN,
        "MINORITY_SHARE": minority_share,
        "MINORITY_CONFIDENCE": minority_confidence,
        "MINORITY_RELATIVE_HALF_WIDTH": half_width,
        "MINORITY_TARGET_HALF_WIDTH": minority_margin_error,
        "COCHRAN_TOTAL_N": pretrain_size,
        "COCHRAN_INFINITE_N": cochran_infinite_n,
        "COCHRAN_FINITE_N": cochran_finite_n,
        "FINITE_POPULATION_FACTOR": 1.0 + (cochran_infinite_n - 1.0) / population_size,
        "USE_FINITE_CORRECTION": use_finite_correction,
        "COCHRAN_MINORITY_N": pretrain_domain_targets[MINORITY_DOMAIN],
        "PRETRAIN_SIZE": pretrain_size,
        "PRETRAIN_VAL_SIZE": pretrain_val_size,
        "PRETRAIN_POOL_SIZE": pretrain_pool_size,
        "FINETUNE_SIZE": finetune_size,
        "FINETUNE_VAL_SIZE": finetune_val_size,
        "FINETUNE_POOL_SIZE": finetune_pool_size,
        "TOTAL_SAMPLED_ROWS": pretrain_pool_size + finetune_pool_size,
        "PRETRAIN_DOMAIN_TARGETS": pretrain_domain_targets,
        "PRETRAIN_VAL_DOMAIN_TARGETS": pretrain_val_domain_targets,
        "FINETUNE_DOMAIN_TARGETS": finetune_domain_targets,
        "FINETUNE_VAL_DOMAIN_TARGETS": finetune_val_domain_targets,
    }


@lru_cache(maxsize=4)
def _population_from_metadata(default=DEFAULT_POPULATION):
    """Row count of the SlimPajama train split, from the Hub when reachable."""
    try:
        from datasets import load_dataset_builder
        split_info = load_dataset_builder("DKYoon/SlimPajama-6B").info.splits.get("train")
        if split_info is not None and split_info.num_examples:
            return split_info.num_examples
    except Exception as exc:
        print(f"Warning: dataset size unavailable ({exc}); using {default:,}.")
    return default



