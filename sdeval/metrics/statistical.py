from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional import guard
    from scipy.spatial.distance import jensenshannon
except Exception:  # pragma: no cover
    jensenshannon = None

try:  # pragma: no cover
    from scipy.stats import chisquare, ks_2samp, wasserstein_distance
except Exception:  # pragma: no cover
    from scipy.stats import wasserstein_distance  # type: ignore
    chisquare = None
    ks_2samp = None

from . import MetricContext, register_metric


def _alpha_beta_metrics(real: pd.DataFrame, syn: pd.DataFrame) -> Dict[str, float]:
    real_cat_cols = [
        col for col in real.columns if real[col].dtype == "object" or str(real[col].dtype).startswith("category")
    ]
    syn_cat_cols = [
        col for col in syn.columns if syn[col].dtype == "object" or str(syn[col].dtype).startswith("category")
    ]
    cat_cols = [col for col in real_cat_cols if col in syn_cat_cols]
    if not cat_cols:
        return {"statistical_alpha_precision": 1.0, "statistical_beta_recall": 1.0}

    alphas: List[float] = []
    betas: List[float] = []
    for col in cat_cols:
        real_vals = set(pd.Series(real[col]).dropna().unique().tolist())
        syn_vals = set(pd.Series(syn[col]).dropna().unique().tolist())
        if not syn_vals or not real_vals:
            continue
        inter = syn_vals & real_vals
        alphas.append(len(inter) / max(1, len(syn_vals)))
        betas.append(len(inter) / max(1, len(real_vals)))

    if not alphas:
        return {"statistical_alpha_precision": 1.0, "statistical_beta_recall": 1.0}

    return {
        "statistical_alpha_precision": float(sum(alphas) / len(alphas)),
        "statistical_beta_recall": float(sum(betas) / len(betas)),
    }


def _numeric_summary_metrics(real: pd.DataFrame, syn: pd.DataFrame) -> Dict[str, float]:
    real_numeric = [col for col in real.columns if np.issubdtype(real[col].dtype, np.number) and col in syn.columns]
    if not real_numeric:
        return {}

    mean_diffs = []
    std_diffs = []
    wasserstein_dists = []
    for col in real_numeric:
        real_col = pd.to_numeric(real[col], errors="coerce").dropna()
        syn_col = pd.to_numeric(syn[col], errors="coerce").dropna()
        if real_col.empty or syn_col.empty:
            continue
        mean_diffs.append(abs(real_col.mean() - syn_col.mean()))
        std_diffs.append(abs(real_col.std(ddof=0) - syn_col.std(ddof=0)))
        if wasserstein_distance is not None:
            wasserstein_dists.append(float(wasserstein_distance(real_col, syn_col)))

    metrics: Dict[str, float] = {}
    if mean_diffs:
        metrics["statistical_mean_abs_mean_diff"] = float(sum(mean_diffs) / len(mean_diffs))
    if std_diffs:
        metrics["statistical_mean_abs_std_diff"] = float(sum(std_diffs) / len(std_diffs))
    if wasserstein_dists:
        metrics["statistical_avg_wasserstein"] = float(sum(wasserstein_dists) / len(wasserstein_dists))
    return metrics


def _ks_metrics(real: pd.DataFrame, syn: pd.DataFrame) -> Dict[str, float]:
    if ks_2samp is None:
        return {}
    numeric_cols = [col for col in real.columns if np.issubdtype(real[col].dtype, np.number) and col in syn.columns]
    if not numeric_cols:
        return {}
    stats: List[float] = []
    for col in numeric_cols:
        r = pd.to_numeric(real[col], errors="coerce").dropna()
        s = pd.to_numeric(syn[col], errors="coerce").dropna()
        if r.empty or s.empty:
            continue
        stat, _ = ks_2samp(r, s)
        stats.append(float(stat))
    if not stats:
        return {}
    return {"statistical_avg_ks": float(sum(stats) / len(stats))}


def _chi_square_metrics(real: pd.DataFrame, syn: pd.DataFrame) -> Dict[str, float]:
    if chisquare is None:
        return {}
    real_cat_cols = [
        col for col in real.columns if real[col].dtype == "object" or str(real[col].dtype).startswith("category")
    ]
    syn_cat_cols = [
        col for col in syn.columns if syn[col].dtype == "object" or str(syn[col].dtype).startswith("category")
    ]
    categorical_cols = [col for col in real_cat_cols if col in syn_cat_cols]
    if not categorical_cols:
        return {}
    stats: List[float] = []
    pvals: List[float] = []
    for col in categorical_cols:
        real_counts = real[col].value_counts(dropna=True)
        syn_counts = syn[col].value_counts(dropna=True)
        if real_counts.empty or syn_counts.empty:
            continue
        categories = real_counts.index.union(syn_counts.index)
        obs = syn_counts.reindex(categories, fill_value=0).astype(float)
        exp = real_counts.reindex(categories, fill_value=0).astype(float)
        total_obs = obs.sum()
        total_exp = exp.sum()
        if total_obs == 0 or total_exp == 0:
            continue
        exp = exp / total_exp * total_obs
        valid = (exp > 0) | (obs > 0)
        if not valid.any():
            continue
        stat, p_val = chisquare(f_obs=obs[valid], f_exp=np.maximum(exp[valid], 1e-9))
        stats.append(float(stat))
        pvals.append(float(p_val))
    if not stats:
        return {}
    metrics = {"statistical_avg_chi2": float(sum(stats) / len(stats))}
    if pvals:
        metrics["statistical_avg_chi2_pvalue"] = float(sum(pvals) / len(pvals))
    return metrics


def _jsd_metrics(real: pd.DataFrame, syn: pd.DataFrame, bins: int = 50) -> Dict[str, float]:
    if jensenshannon is None:
        return {}
    numeric_cols = [col for col in real.columns if np.issubdtype(real[col].dtype, np.number) and col in syn.columns]
    if not numeric_cols:
        return {}
    divergences: List[float] = []
    for col in numeric_cols:
        r = pd.to_numeric(real[col], errors="coerce").dropna()
        s = pd.to_numeric(syn[col], errors="coerce").dropna()
        if r.empty or s.empty:
            continue
        low = min(r.min(), s.min())
        high = max(r.max(), s.max())
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            continue
        hist_range = (low, high)
        r_hist, _ = np.histogram(r, bins=bins, range=hist_range, density=False)
        s_hist, _ = np.histogram(s, bins=bins, range=hist_range, density=False)
        if r_hist.sum() == 0 or s_hist.sum() == 0:
            continue
        r_prob = (r_hist + 1e-12) / (r_hist.sum() + 1e-12 * len(r_hist))
        s_prob = (s_hist + 1e-12) / (s_hist.sum() + 1e-12 * len(s_hist))
        divergence = float(jensenshannon(r_prob, s_prob, base=2.0) ** 2)
        divergences.append(divergence)
    if not divergences:
        return {}
    return {"statistical_avg_jsd": float(sum(divergences) / len(divergences))}


def _correlation_delta_metrics(real: pd.DataFrame, syn: pd.DataFrame) -> Dict[str, float]:
    numeric_cols = [col for col in real.columns if np.issubdtype(real[col].dtype, np.number) and col in syn.columns]
    if len(numeric_cols) < 2:
        return {}
    real_corr = real[numeric_cols].corr().fillna(0.0).to_numpy()
    syn_corr = syn[numeric_cols].corr().fillna(0.0).to_numpy()
    if real_corr.shape != syn_corr.shape:
        return {}
    delta = real_corr - syn_corr
    frob = float(np.linalg.norm(delta, ord="fro"))
    mean_abs = float(np.mean(np.abs(delta)))
    return {
        "statistical_corr_delta_fro": frob,
        "statistical_corr_delta_mean_abs": mean_abs,
    }


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
                                numerical_columns: List[str], categorical_columns: List[str],
                                **kwargs: Any) -> Dict[str, float]:
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
    metrics = {
        'statistical_alpha_precision': compute_alpha_precision(real_df, synthetic_df, categorical_columns),
        'statistical_beta_recall': compute_beta_recall(real_df, synthetic_df, categorical_columns),
        'statistical_mean_abs_mean_diff': compute_mean_absolute_difference(real_df, synthetic_df, numerical_columns),
        'statistical_mean_abs_std_diff': compute_std_absolute_difference(real_df, synthetic_df, numerical_columns),
        'statistical_avg_wasserstein': compute_wasserstein_distance(real_df, synthetic_df, numerical_columns)
    }
    metrics.update(_ks_metrics(real_df, synthetic_df))
    metrics.update(_chi_square_metrics(real_df, synthetic_df))
    metrics.update(_jsd_metrics(real_df, synthetic_df))
    metrics.update(_correlation_delta_metrics(real_df, synthetic_df))
    
    # Remove redundant metrics
    metrics.pop('statistical_avg_chi2', None)  # Keep only p-value
    metrics.pop('statistical_corr_delta_mean_abs', None)  # Keep only Frobenius
    
    return metrics


@register_metric("statistical")
def compute_statistical_metrics_registry(ctx: MetricContext) -> Dict[str, float]:
    """Wrapper for metric registry."""
    real = ctx.real_df
    syn = ctx.synthetic_df

    metrics = {}
    metrics.update(_alpha_beta_metrics(real, syn))
    metrics.update(_numeric_summary_metrics(real, syn))
    metrics.update(_ks_metrics(real, syn))
    metrics.update(_chi_square_metrics(real, syn))
    metrics.update(_jsd_metrics(real, syn))
    metrics.update(_correlation_delta_metrics(real, syn))
    
    # Remove redundant metrics
    metrics.pop('statistical_avg_chi2', None)  # Keep only p-value
    metrics.pop('statistical_corr_delta_mean_abs', None)  # Keep only Frobenius
    
    return metrics
