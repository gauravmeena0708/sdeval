from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from . import MetricContext, register_metric


def _prepare_features(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' missing from DataFrame.")
    X = df.drop(columns=[target])
    y = df[target]
    X = pd.get_dummies(X, dummy_na=True)
    return X, y


@register_metric("ml_efficacy")
def compute_ml_efficacy(ctx: MetricContext) -> Dict[str, float]:
    config = ctx.settings.raw_config or {}
    target = config.get("target_column")
    if not target:
        return {"ml_efficacy_enabled": False, "ml_efficacy_reason": "target_column not set"}

    real_df = ctx.real_df.dropna(subset=[target])
    syn_df = ctx.synthetic_df.dropna(subset=[target])
    if real_df.empty or syn_df.empty:
        return {"ml_efficacy_enabled": False, "ml_efficacy_reason": "insufficient target data"}

    try:
        X_real, y_real = _prepare_features(real_df, target)
        X_syn, y_syn = _prepare_features(syn_df, target)
    except KeyError as exc:
        return {"ml_efficacy_enabled": False, "ml_efficacy_reason": str(exc)}

    X_syn, X_real = X_syn.align(X_real, join="outer", axis=1, fill_value=0)

    task_type = config.get("task_type")
    if not task_type:
        task_type = "classification" if y_real.dtype == "object" or y_real.dtype.name == "category" else "regression"

    results: Dict[str, float] = {"ml_efficacy_enabled": True, "ml_efficacy_task_type": task_type}
    seed = ctx.settings.seed

    if task_type == "classification":
        model = RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1 if hasattr(RandomForestClassifier, "n_jobs") else None
        )
        model.fit(X_syn, y_syn)
        preds = model.predict(X_real)
        results["ml_efficacy_accuracy"] = float(accuracy_score(y_real, preds))
        try:
            results["ml_efficacy_f1_macro"] = float(f1_score(y_real, preds, average="macro"))
        except ValueError:
            results["ml_efficacy_f1_macro"] = float("nan")
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
        model.fit(X_syn, y_syn)
        preds = model.predict(X_real)
        results["ml_efficacy_rmse"] = float(np.sqrt(mean_squared_error(y_real, preds)))
        results["ml_efficacy_mae"] = float(mean_absolute_error(y_real, preds))
        results["ml_efficacy_r2"] = float(r2_score(y_real, preds))

    return results
