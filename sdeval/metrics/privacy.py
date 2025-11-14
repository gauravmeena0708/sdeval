from __future__ import annotations

from typing import Any, Dict, Tuple, List, Optional

import math
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split

from . import MetricContext, register_metric


DEFAULT_DCR_THRESHOLDS = [1e-8, 1e-6, 1e-4, 1e-2]


def _select_numeric_overlap(real: pd.DataFrame, syn: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    num_cols = [col for col in real.columns if np.issubdtype(real[col].dtype, np.number) and col in syn.columns]
    if not num_cols:
        return np.zeros((len(syn), 1), dtype=float), np.zeros((len(real), 1), dtype=float)
    real_data_array = real[num_cols].copy().fillna(0.0).to_numpy(dtype=float)
    synthetic_data_array = syn[num_cols].copy().fillna(0.0).to_numpy(dtype=float)
    return real_data_array, synthetic_data_array


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


def compute_privacy_metrics(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numerical_columns: List[str],
    dcr_threshold: float = 1e-8,
    dcr_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute privacy metrics outside of the MetricContext registry (used by tests/examples).
    """
    if not numerical_columns:
        return {
            "dcr_rate": 0.0,
            "nndr_mean": 0.0,
            "mean_knn_distance": 0.0,
        }

    real_data, syn_data = _prepare_numeric_data(real_df, synthetic_df, numerical_columns)

    if real_data.shape[1] == 0 or len(real_data) == 0 or len(syn_data) == 0:
        return {
            "dcr_rate": 0.0,
            "nndr_mean": 0.0,
            "mean_knn_distance": 0.0,
        }

    n_neighbors = 2 if len(real_data) >= 2 else 1
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(real_data)
    distances, _ = nbrs.kneighbors(syn_data)
    d1 = distances[:, 0]

    metrics = {
        "dcr_rate": float(np.mean(d1 < dcr_threshold)),
        "mean_knn_distance": float(np.mean(d1)),
    }

    if n_neighbors >= 2:
        d2 = distances[:, 1]
        eps = 1e-12
        metrics["nndr_mean"] = float(np.mean(d1 / np.maximum(d2, eps)))
    else:
        metrics["nndr_mean"] = 0.0

    thresholds = _clean_thresholds(dcr_thresholds) or DEFAULT_DCR_THRESHOLDS
    metrics.update(_distance_stats(d1, thresholds))
    return metrics


def _clean_thresholds(values: Optional[List[float]]) -> List[float]:
    if not values:
        return []
    cleaned: List[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric <= 0:
            continue
        cleaned.append(numeric)
    return sorted(set(cleaned))


def _label_threshold(value: float) -> str:
    if value >= 1:
        return str(int(value))
    return f"{value:.0e}".replace("+", "")


def _distance_stats(distances: np.ndarray, thresholds: List[float]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if distances.size == 0:
        return stats
    for threshold in thresholds:
        stats[f"privacy_dcr_at_{_label_threshold(threshold)}"] = float(np.mean(distances < threshold))
    percentiles = [5, 25, 50, 75, 95, 99]
    values = np.percentile(distances, percentiles)
    for p, v in zip(percentiles, values):
        stats[f"privacy_distance_p{p}"] = float(v)
    stats["privacy_distance_mean"] = float(np.mean(distances))
    stats["privacy_distance_std"] = float(np.std(distances))
    return stats


def compute_k_anonymity_metrics(
    synthetic_df: pd.DataFrame,
    quasi_identifiers: List[str],
    k: int,
) -> Dict[str, Any]:
    available_cols = [col for col in quasi_identifiers if col in synthetic_df.columns]
    if not available_cols:
        return {
            "privacy_k_anonymity_enabled": False,
            "privacy_k_anonymity_reason": "quasi identifiers missing from synthetic data",
        }

    filled = synthetic_df[available_cols].fillna("__NA__")
    groups = filled.groupby(available_cols, dropna=False).size()
    total_records = len(synthetic_df)
    if total_records == 0 or groups.empty:
        return {
            "privacy_k_anonymity_enabled": False,
            "privacy_k_anonymity_reason": "synthetic data empty",
        }

    violating = groups[groups < k]
    metrics: Dict[str, Any] = {
        "privacy_k_anonymity_enabled": True,
        "privacy_k_anonymity_k": int(k),
        "privacy_k_anonymity_quasi_columns": available_cols,
        "privacy_k_anonymity_min_group": int(groups.min()),
        "privacy_k_anonymity_total_groups": int(groups.shape[0]),
        "privacy_k_anonymity_violating_groups_ratio": float(len(violating) / len(groups)),
        "privacy_k_anonymity_violating_records_ratio": float(violating.sum() / total_records),
    }
    return metrics


def compute_membership_inference_metrics(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    shared_columns = [col for col in real_df.columns if col in synthetic_df.columns]
    if not shared_columns:
        return {
            "privacy_mia_enabled": False,
            "privacy_mia_reason": "no overlapping columns between real and synthetic data",
        }

    real = real_df[shared_columns].copy()
    syn = synthetic_df[shared_columns].copy()
    if real.empty or syn.empty:
        return {
            "privacy_mia_enabled": False,
            "privacy_mia_reason": "insufficient data for membership inference",
        }

    if sample_size:
        per_class = min(sample_size, len(real), len(syn))
        if per_class < 5:
            return {
                "privacy_mia_enabled": False,
                "privacy_mia_reason": "sample size too small",
            }
        rng = np.random.default_rng(random_state)
        real = real.iloc[rng.choice(len(real), per_class, replace=False)]
        syn = syn.iloc[rng.choice(len(syn), per_class, replace=False)]

    labels = np.concatenate([np.ones(len(real)), np.zeros(len(syn))])
    combined = pd.concat([real, syn], axis=0, ignore_index=True)
    for col in combined.columns:
        if np.issubdtype(combined[col].dtype, np.number):
            combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0)
        else:
            combined[col] = combined[col].astype(str).fillna("__NA__")

    combined = pd.get_dummies(combined, dummy_na=False)
    if combined.shape[1] == 0:
        return {
            "privacy_mia_enabled": False,
            "privacy_mia_reason": "feature matrix empty after encoding",
        }

    if len(np.unique(labels)) < 2:
        return {
            "privacy_mia_enabled": False,
            "privacy_mia_reason": "not enough class variety",
        }

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            combined, labels, test_size=0.3, stratify=labels, random_state=random_state
        )
    except ValueError:
        return {
            "privacy_mia_enabled": False,
            "privacy_mia_reason": "unable to split data for membership inference",
        }

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {
            "privacy_mia_enabled": False,
            "privacy_mia_reason": "not enough class variety after split",
        }

    try:
        model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        accuracy = float(accuracy_score(y_test, preds))
        try:
            auc = float(roc_auc_score(y_test, proba))
        except ValueError:
            auc = float("nan")
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "privacy_mia_enabled": False,
            "privacy_mia_reason": f"membership inference failed: {exc}",
        }

    return {
        "privacy_mia_enabled": True,
        "privacy_mia_accuracy": accuracy,
        "privacy_mia_auc": auc,
        "privacy_mia_samples": int(len(X_train) + len(X_test)),
    }


def _prepare_feature_matrix(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target].copy()
    X = df.drop(columns=[target]).copy()
    if X.empty:
        return pd.DataFrame(), y
    for col in X.columns:
        if np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
        else:
            X[col] = X[col].astype(str).fillna("__NA__")
    X = pd.get_dummies(X, dummy_na=False)
    return X, y


def compute_attribute_inference_metrics(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    targets: List[Dict[str, Any]],
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    if not targets:
        return {}

    metrics: Dict[str, Any] = {}
    for spec in targets:
        column = spec.get("column")
        if not column:
            continue
        prefix = f"privacy_aia_{column}"
        if column not in real_df.columns or column not in synthetic_df.columns:
            metrics[f"{prefix}_enabled"] = False
            metrics[f"{prefix}_reason"] = "column missing from real or synthetic data"
            continue

        real_subset = real_df
        syn_subset = synthetic_df
        if sample_size:
            per_class = min(sample_size, len(real_subset), len(syn_subset))
            if per_class < 20:
                metrics[f"{prefix}_enabled"] = False
                metrics[f"{prefix}_reason"] = "sample size too small"
                continue
            rng = np.random.default_rng(random_state)
            real_subset = real_subset.iloc[rng.choice(len(real_subset), per_class, replace=False)]
            syn_subset = syn_subset.iloc[rng.choice(len(syn_subset), per_class, replace=False)]

        X_train, y_train = _prepare_feature_matrix(syn_subset, column)
        X_test, y_test = _prepare_feature_matrix(real_subset, column)
        if X_train.empty or X_test.empty:
            metrics[f"{prefix}_enabled"] = False
            metrics[f"{prefix}_reason"] = "feature matrix empty after encoding"
            continue
        X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)
        y_train = y_train.dropna()
        y_test = y_test.dropna()
        if y_train.empty or y_test.empty:
            metrics[f"{prefix}_enabled"] = False
            metrics[f"{prefix}_reason"] = "missing target values"
            continue

        task = spec.get("task")
        if not task:
            if y_train.dtype == "object" or y_train.dtype.name == "category" or y_train.nunique() <= 20:
                task = "classification"
            else:
                task = "regression"

        try:
            if task == "classification":
                model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                proba_supported = hasattr(model, "predict_proba")
                metrics[f"{prefix}_accuracy"] = float(accuracy_score(y_test, preds))
                try:
                    metrics[f"{prefix}_f1_macro"] = float(f1_score(y_test, preds, average="macro"))
                except ValueError:
                    metrics[f"{prefix}_f1_macro"] = float("nan")
                if proba_supported and len(set(y_test)) == 2:
                    try:
                        proba = model.predict_proba(X_test)[:, 1]
                        metrics[f"{prefix}_auc"] = float(roc_auc_score(y_test, proba))
                    except ValueError:
                        metrics[f"{prefix}_auc"] = float("nan")
                metrics[f"{prefix}_enabled"] = True
            else:
                model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                metrics[f"{prefix}_rmse"] = float(np.sqrt(mean_squared_error(y_test, preds)))
                metrics[f"{prefix}_mae"] = float(mean_absolute_error(y_test, preds))
                metrics[f"{prefix}_r2"] = float(r2_score(y_test, preds))
                metrics[f"{prefix}_enabled"] = True
        except Exception as exc:  # pragma: no cover - defensive
            metrics[f"{prefix}_enabled"] = False
            metrics[f"{prefix}_reason"] = f"attribute inference failed: {exc}"
            continue

        metrics[f"{prefix}_samples_train"] = int(len(X_train))
        metrics[f"{prefix}_samples_test"] = int(len(X_test))
    return metrics


def compute_dp_metadata_metrics(dp_cfg: Dict[str, Any]) -> Dict[str, Any]:
    epsilon = dp_cfg.get("epsilon")
    if epsilon is None:
        return {
            "privacy_dp_enabled": False,
            "privacy_dp_reason": "epsilon not provided",
        }

    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError):
        return {
            "privacy_dp_enabled": False,
            "privacy_dp_reason": "epsilon invalid",
        }
    if epsilon <= 0:
        return {
            "privacy_dp_enabled": False,
            "privacy_dp_reason": "epsilon must be positive",
        }

    metrics: Dict[str, Any] = {
        "privacy_dp_enabled": True,
        "privacy_dp_epsilon": epsilon,
    }

    if dp_cfg.get("delta") is not None:
        try:
            delta = float(dp_cfg["delta"])
            metrics["privacy_dp_delta"] = delta
        except (TypeError, ValueError):
            metrics["privacy_dp_delta"] = None
    if dp_cfg.get("mechanism"):
        metrics["privacy_dp_mechanism"] = str(dp_cfg["mechanism"])
    if dp_cfg.get("notes"):
        metrics["privacy_dp_notes"] = str(dp_cfg["notes"])
    if dp_cfg.get("training_samples"):
        metrics["privacy_dp_training_samples"] = int(dp_cfg["training_samples"])
    if dp_cfg.get("epochs"):
        metrics["privacy_dp_epochs"] = int(dp_cfg["epochs"])
    if dp_cfg.get("noise_multiplier"):
        try:
            metrics["privacy_dp_noise_multiplier"] = float(dp_cfg["noise_multiplier"])
        except (TypeError, ValueError):
            pass

    # Simple re-identification bound using epsilon
    try:
        reid_bound = 1.0 - math.exp(-epsilon)
        metrics["privacy_dp_reid_bound"] = float(min(max(reid_bound, 0.0), 1.0))
    except OverflowError:
        metrics["privacy_dp_reid_bound"] = 1.0

    if epsilon > dp_cfg.get("warning_threshold", 8):
        metrics["privacy_dp_warning"] = "epsilon is higher than recommended ( > 8 )"

    return metrics


@register_metric("privacy")
def compute_privacy_metrics_registry(ctx: MetricContext) -> Dict[str, float]:
    """Wrapper for metric registry."""
    real = ctx.real_df
    syn = ctx.synthetic_df
    real_data_array, synthetic_data_array = _select_numeric_overlap(real, syn)

    if real_data_array.shape[1] == 0:
        return {"privacy_enabled": False, "privacy_reason": "no shared numeric columns"}

    n_neighbors = 2 if len(real_data_array) >= 2 else 1
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(real_data_array)
    distances, _ = nbrs.kneighbors(synthetic_data_array)

    d1 = distances[:, 0]
    knn_mean = float(np.mean(d1)) if len(d1) else 0.0

    if n_neighbors >= 2:
        d2 = distances[:, 1]
        eps = 1e-12
        nndr_vals = d1 / np.maximum(d2, eps)
        nndr_mean = float(np.mean(nndr_vals))
    else:
        nndr_mean = 0.0

    privacy_cfg = ctx.settings.raw_config.get("privacy_metrics", {}) if hasattr(ctx.settings, "raw_config") else {}
    thresholds = _clean_thresholds(privacy_cfg.get("dcr_thresholds"))
    if not thresholds:
        thresholds = DEFAULT_DCR_THRESHOLDS

    metrics: Dict[str, Any] = {
        "privacy_enabled": True,
        "privacy_dcr": float(np.mean(d1 < 1e-8)),
        "privacy_nndr": nndr_mean,
        "privacy_knn_distance": knn_mean,
    }
    metrics.update(_distance_stats(d1, thresholds))

    k_cfg = privacy_cfg.get("k_anonymity", {})
    quasi_cols = k_cfg.get("quasi_identifiers") or []
    if quasi_cols:
        k_value = int(k_cfg.get("k", 5))
        metrics.update(compute_k_anonymity_metrics(ctx.synthetic_df, quasi_cols, k_value))
    else:
        metrics.setdefault("privacy_k_anonymity_enabled", False)

    mia_cfg = privacy_cfg.get("mia", {})
    if mia_cfg.get("enabled"):
        mia_sample = mia_cfg.get("sample_size")
        try:
            mia_metrics = compute_membership_inference_metrics(
                ctx.real_df,
                ctx.synthetic_df,
                sample_size=mia_sample,
                random_state=ctx.settings.seed or 42,
            )
            metrics.update(mia_metrics)
        except Exception as exc:  # pragma: no cover - defensive
            metrics.update(
                {
                    "privacy_mia_enabled": False,
                    "privacy_mia_reason": f"membership inference failed: {exc}",
                }
            )
    else:
        metrics.setdefault("privacy_mia_enabled", False)

    aia_cfg = privacy_cfg.get("attribute_inference", {})
    aia_targets = aia_cfg.get("targets") or []
    if aia_targets:
        aia_sample = aia_cfg.get("sample_size")
        aia_metrics = compute_attribute_inference_metrics(
            ctx.real_df,
            ctx.synthetic_df,
            aia_targets,
            sample_size=aia_sample,
            random_state=ctx.settings.seed or 42,
        )
        metrics.update(aia_metrics)

    dp_cfg = privacy_cfg.get("dp_metadata")
    if dp_cfg:
        metrics.update(compute_dp_metadata_metrics(dp_cfg))
    else:
        metrics.setdefault("privacy_dp_enabled", False)

    return metrics
