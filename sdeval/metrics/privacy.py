from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import MetricContext, register_metric


def _select_numeric_overlap(real: pd.DataFrame, syn: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    num_cols = [c for c in real.columns if np.issubdtype(real[c].dtype, np.number) and c in syn.columns]
    if not num_cols:
        return np.zeros((len(syn), 1), dtype=float), np.zeros((len(real), 1), dtype=float)
    R = real[num_cols].copy().fillna(0.0).to_numpy(dtype=float)
    G = syn[num_cols].copy().fillna(0.0).to_numpy(dtype=float)
    return R, G


def _prepare_numeric_data(real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                         numerical_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare and standardize numerical data for privacy metrics.

    Args:
        real_df: Real dataset
        synthetic_df: Synthetic dataset
        numerical_columns: List of numerical column names

    Returns:
        Tuple of (real_array, synthetic_array) standardized
    """
    if not numerical_columns:
        return np.zeros((len(real_df), 1)), np.zeros((len(synthetic_df), 1))

    # Select only numerical columns that exist in both datasets
    common_cols = [col for col in numerical_columns
                   if col in real_df.columns and col in synthetic_df.columns]

    if not common_cols:
        return np.zeros((len(real_df), 1)), np.zeros((len(synthetic_df), 1))

    # Extract numerical data and fill NaN with 0
    real_data = real_df[common_cols].fillna(0.0).to_numpy(dtype=float)
    syn_data = synthetic_df[common_cols].fillna(0.0).to_numpy(dtype=float)

    # Standardize features
    scaler = StandardScaler()
    real_data = scaler.fit_transform(real_data)
    syn_data = scaler.transform(syn_data)

    return real_data, syn_data


# Public API functions for testing

def compute_dcr(real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                numerical_columns: List[str], threshold: float = 1e-8) -> float:
    """
    Compute Distance to Closest Record (DCR) rate.

    DCR measures the fraction of synthetic records that are suspiciously close
    to real records, indicating potential privacy leakage.

    Args:
        real_df: Real dataset
        synthetic_df: Synthetic dataset
        numerical_columns: List of numerical column names
        threshold: Distance threshold for considering records "too close"

    Returns:
        DCR rate (fraction of synthetic records below threshold)
    """
    real_data, syn_data = _prepare_numeric_data(real_df, synthetic_df, numerical_columns)

    if real_data.shape[1] == 0 or len(real_data) == 0 or len(syn_data) == 0:
        return 0.0

    # Find nearest neighbor for each synthetic record
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(real_data)
    distances, _ = nbrs.kneighbors(syn_data)

    # Count records below threshold
    close_records = np.sum(distances[:, 0] < threshold)
    return float(close_records / len(syn_data))


def compute_nndr(real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                 numerical_columns: List[str]) -> float:
    """
    Compute Nearest Neighbor Distance Ratio (NNDR).

    NNDR is the ratio of distances to the nearest vs second-nearest real neighbor.
    Lower values indicate synthetic records are suspiciously close to specific real records.

    Args:
        real_df: Real dataset
        synthetic_df: Synthetic dataset
        numerical_columns: List of numerical column names

    Returns:
        Mean NNDR across all synthetic records
    """
    real_data, syn_data = _prepare_numeric_data(real_df, synthetic_df, numerical_columns)

    if real_data.shape[1] == 0 or len(real_data) < 2 or len(syn_data) == 0:
        return 0.0

    # Find 2 nearest neighbors
    n_neighbors = min(2, len(real_data))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(real_data)
    distances, _ = nbrs.kneighbors(syn_data)

    if n_neighbors < 2:
        return 0.0

    d1 = distances[:, 0]
    d2 = distances[:, 1]

    # Compute ratio, avoiding division by zero
    eps = 1e-12
    nndr_vals = d1 / np.maximum(d2, eps)

    return float(np.mean(nndr_vals))


def compute_mean_knn_distance(real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                              numerical_columns: List[str]) -> float:
    """
    Compute mean distance from synthetic to nearest real record.

    Larger distances indicate better privacy (synthetic data is not too similar to real).

    Args:
        real_df: Real dataset
        synthetic_df: Synthetic dataset
        numerical_columns: List of numerical column names

    Returns:
        Mean k-NN distance
    """
    real_data, syn_data = _prepare_numeric_data(real_df, synthetic_df, numerical_columns)

    if real_data.shape[1] == 0 or len(real_data) == 0 or len(syn_data) == 0:
        return 0.0

    # Find nearest neighbor
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(real_data)
    distances, _ = nbrs.kneighbors(syn_data)

    return float(np.mean(distances[:, 0]))


def compute_privacy_metrics(real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                            numerical_columns: List[str], dcr_threshold: float = 1e-8) -> Dict[str, float]:
    """
    Compute all privacy metrics.

    Args:
        real_df: Real dataset
        synthetic_df: Synthetic dataset
        numerical_columns: List of numerical column names
        dcr_threshold: Distance threshold for DCR computation

    Returns:
        Dictionary with all privacy metrics
    """
    return {
        'dcr_rate': compute_dcr(real_df, synthetic_df, numerical_columns, dcr_threshold),
        'nndr_mean': compute_nndr(real_df, synthetic_df, numerical_columns),
        'mean_knn_distance': compute_mean_knn_distance(real_df, synthetic_df, numerical_columns)
    }


@register_metric("privacy")
def compute_privacy_metrics_registry(ctx: MetricContext) -> Dict[str, float]:
    """Wrapper for metric registry."""
    real = ctx.real_df
    syn = ctx.synthetic_df
    R, G = _select_numeric_overlap(real, syn)

    if R.shape[1] == 0:
        return {"privacy_enabled": False, "privacy_reason": "no shared numeric columns"}

    n_neighbors = 2 if len(R) >= 2 else 1
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(R)
    distances, _ = nbrs.kneighbors(G)

    d1 = distances[:, 0]
    knn_mean = float(np.mean(d1)) if len(d1) else 0.0

    if n_neighbors >= 2:
        d2 = distances[:, 1]
        eps = 1e-12
        nndr_vals = d1 / np.maximum(d2, eps)
        nndr_mean = float(np.mean(nndr_vals))
    else:
        nndr_mean = 0.0

    dcr = float(np.mean(d1 < 1e-8)) if len(d1) else 0.0
    return {
        "privacy_enabled": True,
        "privacy_dcr": dcr,
        "privacy_nndr": nndr_mean,
        "privacy_knn_distance": knn_mean,
    }
