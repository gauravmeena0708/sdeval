from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_summary(output_dir: str, synthetic_name: str, metrics: Dict[str, Any]) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_path = Path(output_dir) / f"{synthetic_name}_summary.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    return str(file_path)
