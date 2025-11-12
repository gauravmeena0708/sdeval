# Synthetic Data Evaluator - Basic Plan

## Overview
A CLI tool that evaluates synthetic tabular datasets against real training data using metrics that require only the two CSV files (no additional models, configs, or constraints).

## Scope
**Input Requirements:**
- Training data CSV (real data)
- Synthetic data CSV (generated data)

**Metrics Included:**
Only metrics that can be computed directly from comparing the two datasets without external dependencies.

## CLI Interface
```bash
python -m sdeval.main \
    --training-data-csv-path <real_data.csv> \
    --input-path <synthetic_data.csv> \
    --output-dir <output_dir> \
    [--seed <int>]
```

## Metrics Suite

### 1. Statistical Fidelity
Compare distributions between real and synthetic data:

- **Alpha Precision**: Fraction of synthetic categorical values that exist in real data
- **Beta Recall**: Fraction of real categorical values covered by synthetic data
- **Mean Absolute Difference**: Average difference in column means
- **Std Absolute Difference**: Average difference in column standard deviations
- **Wasserstein Distance**: Distribution distance for numerical columns

### 2. Coverage & Diversity
Measure how well synthetic data covers the real data space:

- **Uniqueness Ratio**: Fraction of unique rows in synthetic data
- **Rare Category Retention**: Fraction of rare real categories (< 5% frequency) present in synthetic
- **Missing Category Ratio**: Fraction of real categories absent in synthetic
- **Missingness Delta**: Difference in null/missing value rates

### 3. Privacy (Basic)
Distance-based privacy metrics:

- **DCR (Distance to Closest Record)**: Rate of synthetic records too close to real records
- **NNDR (Nearest Neighbor Distance Ratio)**: Average ratio of distances to nearest vs second-nearest real neighbor
- **Mean k-NN Distance**: Average distance from synthetic to nearest real neighbor

## Architecture

```
sdeval/
├── main.py              # CLI entry point
├── evaluator.py         # Metric orchestration
├── data_loader.py       # CSV loading
├── metrics/
│   ├── __init__.py      # MetricContext + registry
│   ├── statistical.py   # Statistical fidelity metrics
│   ├── coverage.py      # Coverage & diversity metrics
│   └── privacy.py       # Distance-based privacy metrics
└── reporting/
    └── writer.py        # JSON output writer
```

## Execution Flow

1. **Load Data**: Read both CSVs into pandas DataFrames
2. **Auto-detect Schema**: Identify numerical vs categorical columns
3. **Compute Metrics**: Run all three metric suites
4. **Generate Report**: Output single JSON file with all metrics

## Output Format

```json
{
  "metadata": {
    "training_data": "path/to/train.csv",
    "synthetic_data": "path/to/synthetic.csv",
    "timestamp": "2025-01-12T10:30:00",
    "seed": 42
  },
  "statistical": {
    "alpha_precision": 0.95,
    "beta_recall": 0.92,
    "mean_abs_mean_diff": 0.12,
    "mean_abs_std_diff": 0.08,
    "avg_wasserstein": 0.034
  },
  "coverage": {
    "uniqueness_ratio": 0.98,
    "rare_category_retention": 0.75,
    "missing_category_ratio": 0.05,
    "missingness_delta": 0.02
  },
  "privacy": {
    "dcr_rate": 0.02,
    "nndr_mean": 1.45,
    "mean_knn_distance": 2.34
  }
}
```

## Implementation Notes

### What's Included
- No external model dependencies (no plausibility scoring)
- No configuration files required
- No constraint definitions needed
- No ML utility metrics (no target column required)
- Automatic column type detection
- Minimal setup - just point to two CSV files

### Key Design Decisions
1. **Auto-detection**: Column types (numerical/categorical) detected automatically
2. **No Configuration**: All parameters are defaults or auto-detected
3. **Simple Output**: Single JSON file with flat metric structure
4. **Reproducible**: Optional seed parameter for deterministic results

### Dependencies
- pandas: Data loading and manipulation
- numpy: Numerical computations
- scipy: Wasserstein distance, k-NN calculations
- scikit-learn: Distance metrics for privacy

## Success Criteria
A working tool that:
1. Loads two CSV files without errors
2. Computes all metrics automatically
3. Produces a valid JSON output
4. Handles edge cases (missing values, different column orders, type mismatches)
5. Runs deterministically with seed parameter

## Future Extensions (Not in Basic Version)
- Constraint checking (requires constraint definitions)
- ML utility metrics (requires target column specification)
- Plausibility scoring (requires trained model)
- Differential privacy verification
- Visualization outputs
- Per-column detailed reports
