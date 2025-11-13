from __future__ import annotations

from typing import Dict

from . import MetricContext, register_metric


@register_metric("dp")
def compute_dp_metrics(ctx: MetricContext) -> Dict[str, float]:
    cfg = ctx.settings.raw_config.get("differential_privacy", {})
    if not cfg.get("enabled"):
        return {"dp_enabled": False, "dp_reason": "differential privacy not enabled"}
    epsilon = cfg.get("epsilon")
    delta = cfg.get("delta")
    if epsilon is None or delta is None:
        return {"dp_enabled": False, "dp_reason": "epsilon or delta not specified"}
    return {"dp_enabled": True, "dp_epsilon": float(epsilon), "dp_delta": float(delta)}
