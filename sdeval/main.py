from __future__ import annotations

import argparse
import json
import sys

from .config import load_settings_from_args
from .evaluator import Evaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic Data Evaluator CLI")
    parser.add_argument("--input-path", required=True, help="Synthetic CSV file or directory.")
    parser.add_argument("--training-data-csv-path", required=True, help="Real training data CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for evaluation outputs.")
    parser.add_argument("--model-path", help="Optional plausibility model directory.")
    parser.add_argument("--configs", help="Optional JSON configs file.")
    parser.add_argument("--constraints-path", help="Optional JSON file containing constraint rules.")
    parser.add_argument("--metrics", nargs="+", help="Subset of metrics to run.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings_from_args(args)
        evaluator = Evaluator(settings)
        summaries = evaluator.run()
        print(json.dumps({"summaries": summaries}, indent=2))
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
