from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _round_metrics(obj: Any, decimals: int = 3) -> Any:
    """Recursively round all float values in nested dict/list structures."""
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: _round_metrics(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_round_metrics(item, decimals) for item in obj]
    else:
        return obj


def write_summary(output_dir: str, synthetic_name: str, metrics: Dict[str, Any]) -> str:
    """Write evaluation metrics to JSON with values rounded to 3 decimal places."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_path = Path(output_dir) / f"{synthetic_name}_summary.json"
    
    # Round all numeric values to 3 decimal places
    rounded_metrics = _round_metrics(metrics, decimals=3)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(rounded_metrics, f, indent=2, default=str)
    
    return str(file_path)
