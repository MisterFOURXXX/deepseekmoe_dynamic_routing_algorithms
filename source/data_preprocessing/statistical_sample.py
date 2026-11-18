import math
import scipy.stats as st
from datasets import load_dataset_builder

# ---------------------------------------------------------------------------
# 1D Proportional Stratified Sampling constants (SlimPajama Table 4)
# ---------------------------------------------------------------------------
DOMAIN_SHARES = {
    "RedPajamaCommonCrawl":   0.522,
    "RedPajamaC4":            0.267,
    "RedPajamaGithub":        0.052,
    "RedPajamaArXiv":         0.046,
    "RedPajamaBook":          0.042,
    "RedPajamaWikipedia":     0.038,
    "RedPajamaStackExchange": 0.033,
}

# ---------------------------------------------------------------------------
# Statistical Sample Size Calculation (Cochran's Formula)
# ---------------------------------------------------------------------------
def cochran_sample_size(population_size, p=0.5, margin_of_error=0.01, confidence=0.95):
    """Calculate sample size using Cochran's formula with finite population correction."""
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    n0 = (z**2 * p * (1 - p)) / (margin_of_error**2)
    n = n0 / (1 + (n0 - 1) / population_size)
    return math.ceil(n)

def get_stratified_sizes(minority_confidence=0.99, minority_margin_error=0.0130):
    """
    Dynamically calculates all dataset split sizes and domain targets based on 
    Cochran's formula for the minority stratum (StackExchange).
    """
    # 1. Fetch exact number of rows in the dataset
    print("Fetching SlimPajama dataset metadata...")
    try:
        builder = load_dataset_builder("DKYoon/SlimPajama-6B")
        total_population = builder.info.splits["train"].num_examples
    except Exception as e:
        print(f"Warning: Could not fetch dataset size dynamically ({e}). Defaulting to 5,490,000.")
        total_population = 5_490_000

    print(f"Total population: {total_population:,} rows")

    # 2. Cochran calculation
    minority_domain = "RedPajamaStackExchange"
    minority_pop = total_population * DOMAIN_SHARES[minority_domain]

    cochran_n = cochran_sample_size(
        minority_pop, p=0.5, margin_of_error=minority_margin_error, confidence=minority_confidence
    )

    # 3. Calculate split sizes
    pretrain_size = math.ceil(cochran_n / DOMAIN_SHARES[minority_domain])
    pretrain_val_size = math.floor(pretrain_size * 0.1)
    pretrain_pool_size = pretrain_size + pretrain_val_size

    finetune_size = math.floor(pretrain_size * 0.1)
    finetune_val_size = math.floor(finetune_size * 0.1)
    finetune_pool_size = finetune_size + finetune_val_size

    total_sampled_rows = pretrain_pool_size + finetune_pool_size

    # 4. Calculate domain targets
    pretrain_domain_targets = {d: int(pretrain_size * s) for d, s in DOMAIN_SHARES.items()}
    pretrain_val_domain_targets = {d: int(pretrain_val_size * s) for d, s in DOMAIN_SHARES.items()}

    # Adjust for rounding errors
    pretrain_domain_targets[minority_domain] += pretrain_size - sum(pretrain_domain_targets.values())
    pretrain_val_domain_targets[minority_domain] += pretrain_val_size - sum(pretrain_val_domain_targets.values())

    finetune_domain_targets = {d: int(v * 0.1) for d, v in pretrain_domain_targets.items()}
    finetune_domain_targets[minority_domain] += finetune_size - sum(finetune_domain_targets.values())

    finetune_val_domain_targets = {d: int(v * 0.1) for d, v in pretrain_val_domain_targets.items()}
    finetune_val_domain_targets[minority_domain] += finetune_val_size - sum(finetune_val_domain_targets.values())

    return {
        "TOTAL_POPULATION": total_population,
        "COCHRAN_MINORITY_N": cochran_n,
        "PRETRAIN_SIZE": pretrain_size,
        "PRETRAIN_VAL_SIZE": pretrain_val_size,
        "PRETRAIN_POOL_SIZE": pretrain_pool_size,
        "FINETUNE_SIZE": finetune_size,
        "FINETUNE_VAL_SIZE": finetune_val_size,
        "FINETUNE_POOL_SIZE": finetune_pool_size,
        "TOTAL_SAMPLED_ROWS": total_sampled_rows,
        "PRETRAIN_DOMAIN_TARGETS": pretrain_domain_targets,
        "PRETRAIN_VAL_DOMAIN_TARGETS": pretrain_val_domain_targets,
        "FINETUNE_DOMAIN_TARGETS": finetune_domain_targets,
        "FINETUNE_VAL_DOMAIN_TARGETS": finetune_val_domain_targets,
    }