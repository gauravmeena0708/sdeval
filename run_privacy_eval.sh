#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="configs/privacy_eval.json"
mkdir -p "$(dirname "$CONFIG_PATH")"

cat <<'JSON' > "$CONFIG_PATH"
{
  "privacy_metrics": {
    "dcr_thresholds": [1e-8, 1e-6, 1e-4, 1e-2],
    "k_anonymity": {
      "quasi_identifiers": ["age", "sex", "education"],
      "k": 5
    },
    "mia": {
      "enabled": true,
      "sample_size": 500
    },
    "attribute_inference": {
      "sample_size": 2000,
      "targets": [
        {"column": "income", "task": "classification"},
        {"column": "hours.per.week", "task": "regression"}
      ]
    },
    "dp_metadata": {
      "epsilon": 6.0,
      "delta": 1e-6,
      "mechanism": "DP-SGD",
      "notes": "Example DP certificate"
    }
  }
}
JSON

sdeval evaluate \
  --input-path synthetic/samples_education_1st-4th_0.1.csv \
  --real-data-csv-path datasets/adult/train.csv \
  --output-dir outputs \
  --configs "$CONFIG_PATH" \
  --visualize \
  --html-report
