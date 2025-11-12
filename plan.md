# Synthetic Data Evaluator - Implementation Checklist

## Overview
A CLI tool that evaluates synthetic tabular datasets against real training data using metrics that require only the two CSV files (no additional models, configs, or constraints).

---

## 📋 Implementation Checklist

### Phase 1: Project Setup
- [ ] Create project structure
  - [ ] `sdeval/__init__.py`
  - [ ] `sdeval/main.py`
  - [ ] `sdeval/evaluator.py`
  - [ ] `sdeval/data_loader.py`
  - [ ] `sdeval/metrics/__init__.py`
  - [ ] `sdeval/reporting/__init__.py`
  - [ ] `sdeval/reporting/writer.py`
- [ ] Set up `pyproject.toml` with dependencies
  - [ ] pandas
  - [ ] numpy
  - [ ] scipy
  - [ ] scikit-learn
- [ ] Create `.gitignore` for Python projects

### Phase 2: Core Infrastructure
- [ ] **Data Loader** (`data_loader.py`)
  - [ ] Implement CSV loading function
  - [ ] Add error handling for missing files
  - [ ] Add error handling for malformed CSVs
  - [ ] Validate both DataFrames have data
  - [ ] Auto-detect numerical vs categorical columns

- [ ] **Metric Context** (`metrics/__init__.py`)
  - [ ] Define `MetricContext` dataclass
    - [ ] `real_df: pd.DataFrame`
    - [ ] `synthetic_df: pd.DataFrame`
    - [ ] `numerical_columns: List[str]`
    - [ ] `categorical_columns: List[str]`
    - [ ] `seed: Optional[int]`
  - [ ] Implement metric registry pattern
  - [ ] Create `@register_metric` decorator
  - [ ] Create `get_all_metrics()` function

- [ ] **CLI Interface** (`main.py`)
  - [ ] Add argparse setup
    - [ ] `--training-data-csv-path` (required)
    - [ ] `--input-path` (required)
    - [ ] `--output-dir` (required)
    - [ ] `--seed` (optional)
  - [ ] Validate input arguments
  - [ ] Set random seeds (numpy, random)
  - [ ] Call evaluator
  - [ ] Handle errors and exit codes

- [ ] **Evaluator** (`evaluator.py`)
  - [ ] Implement main evaluation orchestration
  - [ ] Load data via data_loader
  - [ ] Auto-detect column types
  - [ ] Create MetricContext
  - [ ] Execute all registered metrics
  - [ ] Collect results into single dict
  - [ ] Call report writer

- [ ] **Report Writer** (`reporting/writer.py`)
  - [ ] Implement JSON output function
  - [ ] Add metadata section (paths, timestamp, seed)
  - [ ] Ensure output directory exists
  - [ ] Write formatted JSON file
  - [ ] Add error handling for write failures

### Phase 3: Statistical Fidelity Metrics
- [ ] **Create** `metrics/statistical.py`
- [ ] **Implement Alpha Precision**
  - [ ] For each categorical column, compute synthetic values ⊆ real values
  - [ ] Return average across all categorical columns
- [ ] **Implement Beta Recall**
  - [ ] For each categorical column, compute real values ⊆ synthetic values
  - [ ] Return average across all categorical columns
- [ ] **Implement Mean Absolute Difference**
  - [ ] Compute mean for each numerical column in both datasets
  - [ ] Calculate absolute differences
  - [ ] Return average across all numerical columns
- [ ] **Implement Std Absolute Difference**
  - [ ] Compute std for each numerical column in both datasets
  - [ ] Calculate absolute differences
  - [ ] Return average across all numerical columns
- [ ] **Implement Wasserstein Distance**
  - [ ] Use `scipy.stats.wasserstein_distance`
  - [ ] Compute for each numerical column
  - [ ] Return average across all numerical columns
- [ ] **Register metric** with `@register_metric("statistical")`
- [ ] **Return dict** with all 5 metrics

### Phase 4: Coverage & Diversity Metrics
- [ ] **Create** `metrics/coverage.py`
- [ ] **Implement Uniqueness Ratio**
  - [ ] Count unique rows in synthetic data
  - [ ] Divide by total rows
- [ ] **Implement Rare Category Retention**
  - [ ] Identify rare categories in real data (< 5% frequency)
  - [ ] Check how many appear in synthetic data
  - [ ] Return retention rate
- [ ] **Implement Missing Category Ratio**
  - [ ] Get all unique categories from real data
  - [ ] Count how many are missing in synthetic
  - [ ] Return missing fraction
- [ ] **Implement Missingness Delta**
  - [ ] Compute null rate in real data
  - [ ] Compute null rate in synthetic data
  - [ ] Return absolute difference
- [ ] **Register metric** with `@register_metric("coverage")`
- [ ] **Return dict** with all 4 metrics

### Phase 5: Privacy Metrics
- [ ] **Create** `metrics/privacy.py`
- [ ] **Implement k-NN Distance Computation**
  - [ ] Standardize numerical features
  - [ ] Handle categorical features (one-hot encoding)
  - [ ] Use `sklearn.neighbors.NearestNeighbors`
  - [ ] Find k=2 nearest real neighbors for each synthetic row
- [ ] **Implement DCR (Distance to Closest Record)**
  - [ ] Get distance to nearest neighbor (k=1)
  - [ ] Count records below threshold (1e-8)
  - [ ] Return rate
- [ ] **Implement NNDR (Nearest Neighbor Distance Ratio)**
  - [ ] Get distances to k=2 nearest neighbors
  - [ ] Compute ratio d1/d2 for each synthetic row
  - [ ] Return mean ratio
- [ ] **Implement Mean k-NN Distance**
  - [ ] Get distance to nearest neighbor (k=1)
  - [ ] Return mean distance
- [ ] **Register metric** with `@register_metric("privacy")`
- [ ] **Return dict** with all 3 metrics

### Phase 6: Integration & Testing
- [ ] **End-to-end test**
  - [ ] Run with Adult dataset
  - [ ] Verify JSON output is created
  - [ ] Verify all metrics are present
  - [ ] Check for errors/warnings
- [ ] **Edge case testing**
  - [ ] Test with missing values
  - [ ] Test with all-categorical dataset
  - [ ] Test with all-numerical dataset
  - [ ] Test with single-column dataset
  - [ ] Test with mismatched columns
- [ ] **Reproducibility test**
  - [ ] Run twice with same seed
  - [ ] Verify identical outputs

### Phase 7: Documentation
- [ ] Create `README.md`
  - [ ] Installation instructions
  - [ ] Quick start example
  - [ ] CLI usage
  - [ ] Metric explanations
- [ ] Add docstrings to all functions
- [ ] Add inline comments for complex logic
- [ ] Create example output JSON

---

## 🎯 Success Criteria

- [ ] CLI runs without errors: `python -m sdeval.main --training-data-csv-path train.csv --input-path synthetic.csv --output-dir outputs/`
- [ ] All 12 metrics computed and present in output
- [ ] JSON output is valid and well-formatted
- [ ] Reproducible results with `--seed` parameter
- [ ] Handles common edge cases gracefully

---

## 📦 Expected Output Structure

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

---

## 🚫 Out of Scope (Future Extensions)

- k-anonymity (requires config for quasi-identifiers)
- Constraint checking (requires constraint definitions)
- ML utility metrics (requires target column specification)
- Plausibility scoring (requires trained model)
- Differential privacy verification
- Visualization outputs
- Per-column detailed reports
