from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from . import MetricContext, register_metric


def _alpha_beta_metrics(real: pd.DataFrame, syn: pd.DataFrame) -> Dict[str, float]:
    real_cat_cols = [
        c for c in real.columns if real[c].dtype == "object" or str(real[c].dtype).startswith("category")
    ]
    syn_cat_cols = [
        c for c in syn.columns if syn[c].dtype == "object" or str(syn[c].dtype).startswith("category")
    ]
    cat_cols = [c for c in real_cat_cols if c in syn_cat_cols]
    if not cat_cols:
        return {"alpha_precision": 1.0, "beta_recall": 1.0}

    alphas: List[float] = []
    betas: List[float] = []
    for c in cat_cols:
        real_vals = set(pd.Series(real[c]).dropna().unique().tolist())
        syn_vals = set(pd.Series(syn[c]).dropna().unique().tolist())
        if not syn_vals or not real_vals:
            continue
        inter = syn_vals & real_vals
        alphas.append(len(inter) / max(1, len(syn_vals)))
        betas.append(len(inter) / max(1, len(real_vals)))

    if not alphas:
        return {"alpha_precision": 1.0, "beta_recall": 1.0}

    return {
        "alpha_precision": float(sum(alphas) / len(alphas)),
        "beta_recall": float(sum(betas) / len(betas)),
    }


def _numeric_summary_metrics(real: pd.DataFrame, syn: pd.DataFrame) -> Dict[str, float]:
    real_numeric = [c for c in real.columns if np.issubdtype(real[c].dtype, np.number) and c in syn.columns]
    if not real_numeric:
        return {}

    mean_diffs = []
    std_diffs = []
    wass = []
    for col in real_numeric:
        real_col = pd.to_numeric(real[col], errors="coerce").dropna()
        syn_col = pd.to_numeric(syn[col], errors="coerce").dropna()
        if real_col.empty or syn_col.empty:
            continue
        mean_diffs.append(abs(real_col.mean() - syn_col.mean()))
        std_diffs.append(abs(real_col.std(ddof=0) - syn_col.std(ddof=0)))
        if wasserstein_distance is not None:
            wass.append(float(wasserstein_distance(real_col, syn_col)))

    metrics: Dict[str, float] = {}
    if mean_diffs:
        metrics["mean_abs_mean_diff"] = float(sum(mean_diffs) / len(mean_diffs))
    if std_diffs:
        metrics["mean_abs_std_diff"] = float(sum(std_diffs) / len(std_diffs))
    if wass:
        metrics["avg_wasserstein"] = float(sum(wass) / len(wass))
    return metrics


# Public API functions for testing

def compute_alpha_precision(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, categorical_columns: List[str]) -> float:
    """
    Compute alpha precision: fraction of synthetic categorical values present in real data.

    Args:
        real_df: Real dataset
        synthetic_df: Synthetic dataset
        categorical_columns: List of categorical column names

    Returns:
        Alpha precision score (0.0 to 1.0)
    """
    if not categorical_columns:
        return 1.0

    alphas = []
    for col in categorical_columns:
        if col not in real_df.columns or col not in synthetic_df.columns:
            continue

        real_vals = set(pd.Series(real_df[col]).dropna().unique())
        syn_vals = set(pd.Series(synthetic_df[col]).dropna().unique())

        if not syn_vals:
            continue

        intersection = syn_vals & real_vals
        alphas.append(len(intersection) / len(syn_vals))

    return float(np.mean(alphas)) if alphas else 1.0


def compute_beta_recall(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, categorical_columns: List[str]) -> float:
    """
    Compute beta recall: fraction of real categorical values covered by synthetic data.

    Args:
        real_df: Real dataset
        synthetic_df: Synthetic dataset
        categorical_columns: List of categorical column names

    Returns:
        Beta recall score (0.0 to 1.0)
    """
    if not categorical_columns:
        return 1.0

    betas = []
    for col in categorical_columns:
        if col not in real_df.columns or col not in synthetic_df.columns:
            continue

        real_vals = set(pd.Series(real_df[col]).dropna().unique())
        syn_vals = set(pd.Series(synthetic_df[col]).dropna().unique())

        if not real_vals:
            continue

        intersection = syn_vals & real_vals
        betas.append(len(intersection) / len(real_vals))

    return float(np.mean(betas)) if betas else 1.0


def compute_mean_absolute_difference(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, numerical_columns: List[str]) -> float:
    """
    Compute average absolute difference in column means.

    Args:
        real_df: Real dataset
        synthetic_df: Synthetic dataset
        numerical_columns: List of numerical column names

    Returns:
        Mean absolute difference
    """
    if not numerical_columns:
        return 0.0

    diffs = []
    for col in numerical_columns:
        if col not in real_df.columns or col not in synthetic_df.columns:
            continue

        real_col = pd.to_numeric(real_df[col], errors='coerce').dropna()
        syn_col = pd.to_numeric(synthetic_df[col], errors='coerce').dropna()

        if real_col.empty or syn_col.empty:
            continue

        diffs.append(abs(real_col.mean() - syn_col.mean()))

    return float(np.mean(diffs)) if diffs else 0.0


def compute_std_absolute_difference(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, numerical_columns: List[str]) -> float:
    """
    Compute average absolute difference in column standard deviations.

    Args:
        real_df: Real dataset
        synthetic_df: Synthetic dataset
        numerical_columns: List of numerical column names

    Returns:
        Std absolute difference
    """
    if not numerical_columns:
        return 0.0

    diffs = []
    for col in numerical_columns:
        if col not in real_df.columns or col not in synthetic_df.columns:
            continue

        real_col = pd.to_numeric(real_df[col], errors='coerce').dropna()
        syn_col = pd.to_numeric(synthetic_df[col], errors='coerce').dropna()

        if real_col.empty or syn_col.empty:
            continue

        diffs.append(abs(real_col.std(ddof=0) - syn_col.std(ddof=0)))

    return float(np.mean(diffs)) if diffs else 0.0


def compute_wasserstein_distance(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, numerical_columns: List[str]) -> float:
    """
    Compute average Wasserstein distance for numerical columns.

    Args:
        real_df: Real dataset
        synthetic_df: Synthetic dataset
        numerical_columns: List of numerical column names

    Returns:
        Average Wasserstein distance
    """
    if not numerical_columns:
        return 0.0

    distances = []
    for col in numerical_columns:
        if col not in real_df.columns or col not in synthetic_df.columns:
            continue

        real_col = pd.to_numeric(real_df[col], errors='coerce').dropna()
        syn_col = pd.to_numeric(synthetic_df[col], errors='coerce').dropna()

        if real_col.empty or syn_col.empty:
            continue

        distances.append(wasserstein_distance(real_col.values, syn_col.values))

    return float(np.mean(distances)) if distances else 0.0


def compute_statistical_metrics(real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                                numerical_columns: List[str], categorical_columns: List[str]) -> Dict[str, float]:
    """
    Compute all statistical fidelity metrics.

    Args:
        real_df: Real dataset
        synthetic_df: Synthetic dataset
        numerical_columns: List of numerical column names
        categorical_columns: List of categorical column names

    Returns:
        Dictionary with all statistical metrics
    """
    return {
        'alpha_precision': compute_alpha_precision(real_df, synthetic_df, categorical_columns),
        'beta_recall': compute_beta_recall(real_df, synthetic_df, categorical_columns),
        'mean_abs_mean_diff': compute_mean_absolute_difference(real_df, synthetic_df, numerical_columns),
        'mean_abs_std_diff': compute_std_absolute_difference(real_df, synthetic_df, numerical_columns),
        'avg_wasserstein': compute_wasserstein_distance(real_df, synthetic_df, numerical_columns)
    }


@register_metric("statistical")
def compute_statistical_metrics_registry(ctx: MetricContext) -> Dict[str, float]:
    """Wrapper for metric registry."""
    real = ctx.real_df
    syn = ctx.synthetic_df

    metrics = {}
    metrics.update(_alpha_beta_metrics(real, syn))
    metrics.update(_numeric_summary_metrics(real, syn))
    return metrics
