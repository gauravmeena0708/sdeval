from __future__ import annotations

from typing import Dict, List

import pandas as pd

from . import MetricContext, register_metric


def _uniqueness_metric(synthetic_df: pd.DataFrame) -> Dict[str, float]:
    n = len(synthetic_df)
    if n == 0:
        return {"coverage_unique_ratio": 0.0}
    unique_rows = synthetic_df.drop_duplicates().shape[0]
    return {"coverage_unique_ratio": float(unique_rows / n)}


def _rare_category_retention(real: pd.DataFrame, syn: pd.DataFrame) -> Dict[str, float]:
    real_cat_cols = [
        col for col in real.columns if real[col].dtype == "object" or str(real[col].dtype).startswith("category")
    ]
    syn_cat_cols = [
        col for col in syn.columns if syn[col].dtype == "object" or str(syn[col].dtype).startswith("category")
    ]
    shared_cols = [col for col in real_cat_cols if col in syn_cat_cols]
    if not shared_cols:
        return {}

    retention: List[float] = []
    missing_support: List[float] = []
    for col in shared_cols:
        real_counts = real[col].value_counts(normalize=True, dropna=True)
        syn_values = set(pd.Series(syn[col]).dropna().unique().tolist())
        rare_categories = real_counts[real_counts <= 0.05].index.tolist()
        if not rare_categories:
            rare_categories = real_counts.tail(min(5, len(real_counts))).index.tolist()
        if not rare_categories:
            continue
        covered = sum(1 for cat in rare_categories if cat in syn_values)
        retention.append(covered / len(rare_categories))
        missing_support.append((len(real_counts.index.difference(syn_values)) / max(1, len(real_counts))))

    metrics: Dict[str, float] = {}
    if retention:
        metrics["coverage_rare_category_retention"] = float(sum(retention) / len(retention))
    if missing_support:
        metrics["coverage_missing_category_ratio"] = float(sum(missing_support) / len(missing_support))
    return metrics


def _missingness_alignment(real: pd.DataFrame, syn: pd.DataFrame) -> Dict[str, float]:
    real_missing = real.isna().mean().mean()
    syn_missing = syn.isna().mean().mean()
    return {
        "coverage_missingness_delta": float(syn_missing - real_missing),
    }


# Public API functions for testing

def compute_uniqueness_ratio(synthetic_df: pd.DataFrame) -> float:
    """
    Compute the fraction of unique rows in the dataset.

    Args:
        synthetic_df: Input DataFrame

    Returns:
        Uniqueness ratio (0.0 to 1.0)
    """
    n = len(synthetic_df)
    if n == 0:
        return 0.0
    unique_rows = synthetic_df.drop_duplicates().shape[0]
    return float(unique_rows / n)


def compute_rare_category_retention(real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                                    categorical_columns: List[str], threshold: float = 0.05) -> float:
    """
    Compute fraction of rare categories (< threshold frequency) from real data that appear in synthetic data.

    Args:
        real_df: Real dataset
        synthetic_df: Synthetic dataset
        categorical_columns: List of categorical column names
        threshold: Frequency threshold for rare categories (default 5%)

    Returns:
        Rare category retention rate (0.0 to 1.0)
    """
    if not categorical_columns:
        return 1.0

    retention = []
    for col in categorical_columns:
        if col not in real_df.columns or col not in synthetic_df.columns:
            continue

        real_counts = real_df[col].value_counts(normalize=True, dropna=True)
        syn_values = set(pd.Series(synthetic_df[col]).dropna().unique())

        # Find rare categories
        rare_categories = real_counts[real_counts <= threshold].index.tolist()

        if not rare_categories:
            continue

        # Count how many rare categories are retained
        covered = sum(1 for cat in rare_categories if cat in syn_values)
        retention.append(covered / len(rare_categories))

    return float(sum(retention) / len(retention)) if retention else 1.0


def compute_missing_category_ratio(real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                                   categorical_columns: List[str]) -> float:
    """
    Compute fraction of real categories that are missing in synthetic data.

    Args:
        real_df: Real dataset
        synthetic_df: Synthetic dataset
        categorical_columns: List of categorical column names

    Returns:
        Missing category ratio (0.0 to 1.0)
    """
    if not categorical_columns:
        return 0.0

    missing_ratios = []
    for col in categorical_columns:
        if col not in real_df.columns or col not in synthetic_df.columns:
            continue

        real_values = set(pd.Series(real_df[col]).dropna().unique())
        syn_values = set(pd.Series(synthetic_df[col]).dropna().unique())

        if not real_values:
            continue

        missing = len(real_values - syn_values)
        missing_ratios.append(missing / len(real_values))

    return float(sum(missing_ratios) / len(missing_ratios)) if missing_ratios else 0.0


def compute_missingness_delta(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
    """
    Compute absolute difference in overall null/missing rates between datasets.

    Args:
        real_df: Real dataset
        synthetic_df: Synthetic dataset

    Returns:
        Absolute difference in missingness rates
    """
    real_missing = real_df.isna().mean().mean()
    syn_missing = synthetic_df.isna().mean().mean()
    return float(abs(syn_missing - real_missing))


def compute_coverage_metrics(real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                             categorical_columns: List[str]) -> Dict[str, float]:
    """
    Compute all coverage and diversity metrics.

    Args:
        real_df: Real dataset
        synthetic_df: Synthetic dataset
        categorical_columns: List of categorical column names

    Returns:
        Dictionary with all coverage metrics
    """
    return {
        'coverage_uniqueness_ratio': compute_uniqueness_ratio(synthetic_df),
        'coverage_rare_category_retention': compute_rare_category_retention(real_df, synthetic_df, categorical_columns),
        'coverage_missing_category_ratio': compute_missing_category_ratio(real_df, synthetic_df, categorical_columns),
        'coverage_missingness_delta': compute_missingness_delta(real_df, synthetic_df)
    }


@register_metric("coverage")
def compute_coverage_metrics_registry(ctx: MetricContext) -> Dict[str, float]:
    """Wrapper for metric registry."""
    real = ctx.real_df
    syn = ctx.synthetic_df

    metrics = {}
    metrics.update(_uniqueness_metric(syn))
    metrics.update(_rare_category_retention(real, syn))
    metrics.update(_missingness_alignment(real, syn))
    return metrics
