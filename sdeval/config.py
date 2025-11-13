from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class EvaluatorSettings:
    """Runtime settings derived from CLI arguments and optional config files."""

    input_path: str
    real_data_path: str
    output_dir: str
    model_path: Optional[str] = None
    overwrite: bool = False
    seed: Optional[int] = None
    raw_config: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metrics: List[str] = field(default_factory=list)

    def ensure_output_dir(self) -> None:
        path = Path(self.output_dir)
        if path.exists() and not self.overwrite:
            return
        path.mkdir(parents=True, exist_ok=True)


def load_settings_from_args(args) -> EvaluatorSettings:
    """Merge CLI args, configs JSON, and constraints JSON into EvaluatorSettings."""

    config_dict = _load_json(args.configs)
    constraints_dict = _load_json(args.constraints_path)

    # Allow constraint definitions to live within the general config.
    if not constraints_dict:
        constraints_dict = config_dict.get("constraints", {})
    elif config_dict.get("constraints"):
        # Merge explicit constraints on top of config-provided ones.
        merged = dict(config_dict.get("constraints", {}))
        merged_rules = list(merged.get("rules", []))
        merged_rules.extend(constraints_dict.get("rules", []))
        merged["rules"] = merged_rules
        constraints_dict = merged

    metrics = args.metrics or config_dict.get("metrics") or [
        "statistical",
        "constraints",
        "coverage",
        "ml_efficacy",
        "plausibility",
        "privacy",
        "dp",
    ]

    output_dir = args.output_dir or config_dict.get("output_dir") or "outputs"

    settings = EvaluatorSettings(
        input_path=args.input_path,
        real_data_path=args.real_data_csv_path or config_dict.get("real_data_csv_path"),
        output_dir=output_dir,
        model_path=args.model_path or config_dict.get("model_path"),
        overwrite=args.overwrite,
        seed=args.seed or config_dict.get("seed"),
        raw_config=config_dict,
        constraints=constraints_dict,
        metrics=metrics,
    )

    if not settings.real_data_path:
        raise ValueError("Real data CSV path must be provided via CLI or configs.")
    if not os.path.exists(settings.real_data_path):
        raise FileNotFoundError(f"Real data CSV not found: {settings.real_data_path}")
    if not os.path.exists(settings.input_path):
        raise FileNotFoundError(f"Synthetic input path not found: {settings.input_path}")

    settings.ensure_output_dir()
    return settings
