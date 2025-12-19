from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import MetricContext, register_metric


# ============================================================================
# Simple Constraint Satisfaction Rate (for categorical constraints)
# ============================================================================

def parse_constraint(constraint: Optional[str]) -> List[Tuple[str, str]]:
    """
    Parse a simple categorical constraint string into list of (column, value) tuples.

    Supports:
    - Single constraint: "education=11th"
    - Multiple constraints: "workclass=State-gov,education=Bachelors"

    Args:
        constraint: Constraint string (e.g., "education=11th" or "col1=val1,col2=val2")

    Returns:
        List of (column, value) tuples

    Examples:
        >>> parse_constraint("education=11th")
        [('education', '11th')]
        >>> parse_constraint("workclass=State-gov,education=Bachelors")
        [('workclass', 'State-gov'), ('education', 'Bachelors')]
    """
    if not constraint:
        return []

    parsed = []
    # Split by comma for multiple constraints
    parts = constraint.split(',')
    for part in parts:
        part = part.strip()
        if '=' not in part:
            continue
        # Split by = for column=value
        column, value = part.split('=', 1)
        column = column.strip()
        value = value.strip()
        parsed.append((column, value))

    return parsed


def compute_constraint_satisfaction_rate(real_df: pd.DataFrame, constraint: str) -> float:
    """
    Compute the proportion of samples satisfying a categorical constraint.

    Args:
        real_df: DataFrame to evaluate
        constraint: Constraint string (e.g., "education=11th" or "col1=val1,col2=val2")

    Returns:
        Float between 0 and 1 representing the satisfaction rate

    Examples:
        >>> real_df = pd.DataFrame({'education': ['11th', 'Bachelors', '11th']})
        >>> compute_constraint_satisfaction_rate(real_df, "education=11th")
        0.6666666666666666
    """
    if real_df.empty:
        return 0.0

    # Parse the constraint
    constraints = parse_constraint(constraint)

    # Empty constraint means all rows satisfy (vacuous truth)
    if not constraints:
        return 1.0

    # Start with all rows as True
    mask = pd.Series([True] * len(real_df), index=real_df.index)

    # Apply each constraint with AND logic
    for column, value in constraints:
        if column not in real_df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame. Available columns: {list(real_df.columns)}")

        # Convert to string and strip whitespace for comparison
        # This handles datasets with leading/trailing spaces
        mask &= (real_df[column].astype(str).str.strip() == value)

    # Calculate satisfaction rate
    satisfaction_rate = mask.sum() / len(real_df)
    return float(satisfaction_rate)


def compute_constraint_support(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    constraint: str
) -> Dict[str, float]:
    """
    Compute constraint satisfaction/support metrics for both real and synthetic data.

    Args:
        real_df: Real data DataFrame
        synthetic_df: Synthetic data DataFrame
        constraint: Constraint string (e.g., "education=11th")

    Returns:
        Dictionary with:
        - real_satisfaction_rate: Satisfaction rate in real data
        - synthetic_satisfaction_rate: Satisfaction rate in synthetic data
        - satisfaction_rate_diff: Absolute difference between rates

    Examples:
        >>> real = pd.DataFrame({'education': ['11th', 'Bachelors', '11th']})
        >>> synth = pd.DataFrame({'education': ['11th', '11th', 'Masters']})
        >>> metrics = compute_constraint_support(real, synth, "education=11th")
        >>> metrics['real_satisfaction_rate']
        0.6666666666666666
        >>> metrics['synthetic_satisfaction_rate']
        0.6666666666666666
    """
    real_rate = compute_constraint_satisfaction_rate(real_df, constraint)
    synthetic_rate = compute_constraint_satisfaction_rate(synthetic_df, constraint)

    return {
        'real_satisfaction_rate': real_rate,
        'synthetic_satisfaction_rate': synthetic_rate,
        'satisfaction_rate_diff': abs(real_rate - synthetic_rate)
    }


# ============================================================================
# Original Complex Constraint Validation (requires configuration)
# ============================================================================


def _build_value_lookup(real_df: pd.DataFrame) -> Dict[tuple, Any]:
    lookup: Dict[tuple, Any] = {}
    for col in real_df.columns:
        series = real_df[col]
        if series.dtype == "object" or str(series.dtype).startswith("category"):
            for val in pd.Series(series).dropna().unique().tolist():
                lookup[(col, str(val).strip())] = val
    return lookup


def _quote_value(val: Any) -> str:
    val = str(val)
    if '"' in val and "'" in val:
        escaped = val.replace('"', '\\"')
        return f'"{escaped}"'
    if '"' in val:
        return f"'{val}'"
    return f'"{val}"'


def _normalize_constraint_expr(expr: str, value_lookup: Dict[tuple, Any]) -> str:
    if not expr:
        return expr

    def repl(match):
        col = match.group("col").strip()
        val_raw = match.group("val").strip()
        val = value_lookup.get((col, val_raw), val_raw)
        return f"`{col}` == {_quote_value(val)}"

    pattern = re.compile(r"(?P<col>[A-Za-z0-9_.]+)\s*=\s*(?P<val>[^&|,]+)")
    return re.sub(pattern, repl, expr)


def _apply_expression(rule: Dict[str, Any], df: pd.DataFrame, value_lookup: Dict[tuple, Any]) -> Dict[str, Any]:
    expr = rule.get("expression")
    if not expr:
        column = rule.get("column")
        value = rule.get("value")
        if column is None or value is None:
            raise ValueError(f"Constraint rule {rule.get('id')} missing column/value.")
        expr = f"{column} = {value}"
    normalized = _normalize_constraint_expr(expr, value_lookup)
    mask = df.eval(normalized)
    satisfied = int(mask.sum())
    total = len(df)
    ratio = satisfied / total if total else 0.0
    target_pct = rule.get("target_pct", 1.0 if rule.get("type") in {"equality", "expression"} else None)
    tolerance_pct = rule.get("tolerance_pct", 0.0)
    deviation = ratio - target_pct if target_pct is not None else None
    hard = rule.get("hard", rule.get("type") in {"equality", "expression"})
    passed = True
    if hard and target_pct == 1.0 and tolerance_pct == 0:
        passed = ratio == 1.0
    elif target_pct is not None:
        passed = abs(deviation) <= tolerance_pct

    return {
        "rule_id": rule.get("id"),
        "type": rule.get("type", "expression"),
        "observed_ratio": ratio,
        "target_ratio": target_pct,
        "deviation": deviation,
        "satisfied_count": satisfied,
        "total_count": total,
        "hard": hard,
        "passed": passed,
    }


def _apply_mean(rule: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    column = rule.get("column")
    target = rule.get("target")
    tolerance = rule.get("tolerance", 0.0)
    if column is None or target is None:
        raise ValueError(f"Mean rule {rule.get('id')} missing column/target.")
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    observed = float(series.mean()) if not series.empty else math.nan
    deviation = observed - target if not math.isnan(observed) else None
    passed = False if deviation is None else abs(deviation) <= tolerance
    return {
        "rule_id": rule.get("id"),
        "type": "mean",
        "observed_value": observed,
        "target_value": target,
        "deviation": deviation,
        "tolerance": tolerance,
        "hard": rule.get("hard", False),
        "passed": passed,
    }


def _apply_bound(rule: Dict[str, Any], df: pd.DataFrame, bound_type: str) -> Dict[str, Any]:
    column = rule.get("column")
    bound_key = "upper_bound" if bound_type == "max" else "lower_bound"
    bound_val = rule.get(bound_key)
    if column is None or bound_val is None:
        raise ValueError(f"{bound_type} rule {rule.get('id')} missing column/{bound_key}.")

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    observed = float(series.max()) if bound_type == "max" else float(series.min())
    passed = observed <= bound_val if bound_type == "max" else observed >= bound_val
    tolerance = rule.get("tolerance", 0.0)
    if tolerance:
        if bound_type == "max":
            passed = observed <= (bound_val + tolerance)
        else:
            passed = observed >= (bound_val - tolerance)
    return {
        "rule_id": rule.get("id"),
        "type": bound_type,
        "observed_value": observed,
        bound_key: bound_val,
        "tolerance": tolerance,
        "hard": rule.get("hard", True),
        "passed": passed,
    }


RULE_HANDLERS = {
    "expression": _apply_expression,
    "equality": _apply_expression,
    "share": _apply_expression,
    "mean": _apply_mean,
    "max": lambda rule, df, lookup: _apply_bound(rule, df, "max"),
    "min": lambda rule, df, lookup: _apply_bound(rule, df, "min"),
}


@register_metric("constraints")
def compute_constraints_metrics(ctx: MetricContext) -> Dict[str, Any]:
    constraints_cfg = ctx.settings.constraints or {}
    rules = constraints_cfg.get("rules", [])
    if not rules:
        return {"constraints_enabled": False}

    value_lookup = _build_value_lookup(ctx.real_df)

    results: List[Dict[str, Any]] = []
    hard_failures = 0
    for rule in rules:
        rule_type = rule.get("type", "expression")
        handler = RULE_HANDLERS.get(rule_type)
        if not handler:
            results.append(
                {
                    "rule_id": rule.get("id"),
                    "type": rule_type,
                    "passed": False,
                    "error": f"Unsupported rule type {rule_type}",
                }
            )
            hard_failures += 1
            continue
        try:
            outcome = handler(rule, ctx.synthetic_df, value_lookup)
        except Exception as exc:  # pragma: no cover - defensive
            outcome = {
                "rule_id": rule.get("id"),
                "type": rule_type,
                "passed": False,
                "error": str(exc),
            }
        results.append(outcome)
        if outcome.get("hard") and not outcome.get("passed"):
            hard_failures += 1

    total_rules = len(results)
    passed_rules = sum(1 for r in results if r.get("passed"))
    return {
        "constraints_enabled": True,
        "constraints_total_rules": total_rules,
        "constraints_passed_rules": passed_rules,
        "constraints_hard_failures": hard_failures,
        "constraints_rule_details": results,
    }
