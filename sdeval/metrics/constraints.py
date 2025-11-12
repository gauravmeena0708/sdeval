from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import MetricContext, register_metric


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
def compute_constraint_metrics(ctx: MetricContext) -> Dict[str, Any]:
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
        "constraint_rule_details": results,
    }
