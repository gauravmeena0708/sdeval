# Synthetic Data Evaluator Plan

## 1. Goals & Scope
- Deliver a CLI tool that scores synthetic tabular datasets against real (training) data while honoring user-supplied constraints (e.g., `COL = VAL`, `mean(COL) = N`, `COL <= N`, population share targets).
- Provide actionable metrics across fidelity, constraint adherence, coverage/diversity, downstream ML utility, plausibility, and privacy.
- Support CSV inputs (single file or directory) and produce machine-readable reports plus human-friendly summaries.
- Non-goals: building/ training synthetic generators, guaranteeing formal differential privacy for every model (placeholder only), or supporting unstructured data.

## 2. Success Criteria
- **Constraint Satisfaction:** Report rate and deviation for every declared rule; flag hard violations (0 tolerance) separately from soft targets (means/shares).
- **Fidelity & Coverage:** Capture 1D/2D distribution alignment, correlations, rare-category preservation, and higher-order dependencies.
- **Utility:** Demonstrate that models trained on synthetic data perform comparably when evaluated on real holdout data.
- **Privacy:** Provide proximity-based leakage checks (DCR) and hooks for more advanced attacks when enabled.
- **Plausibility:** Score records with an autoregressive model when provided.
- **Reproducibility:** Deterministic runs when the same configs and seeds are supplied; outputs stored under `--output-dir` with metadata.

## 3. CLI Surface
```bash
python -m evaluator.main \
    --input-path <synthetic_csv_or_folder> \
    --training-data-csv-path <real_data_csv> \
    --output-dir <output_dir> \
    [--model-path <plausibility_model_dir>] \
    [--configs <configs_json>] \
    [--constraints-path <constraints_json>] \
    [--overwrite] \
    [--seed <int>]
```
- `--input-path`: synthetic data source; directory iteration is recursive (CSV only).
- `--training-data-csv-path`: required reference data.
- `--output-dir`: results root; tool creates subfolders per run timestamp unless `--overwrite` is set.
- `--model-path`: required for plausibility metrics.
- `--configs`: JSON describing schema, target column, constraint rules, metric toggles, categorical encodings, etc.
- `--constraints-path`: optional standalone JSON defining constraint rules/support targets when not bundled inside `--configs`.
- `--seed`: ensures reproducible sampling/model training.

## 4. Configuration Contract (`configs.json`)
Key fields (all optional unless noted):
```json
{
  "target_column": "income",
  "task_type": "classification",             // or "regression"
  "categorical_columns": ["workclass", "education"],
  "numerical_columns": ["age", "fnlwgt", "hours-per-week"],
  "constraints": {
    "rules": [
      {"id": "eq_occupation_private", "type": "equality", "column": "occupation", "value": "Private"},
      {"id": "mean_age", "type": "mean", "column": "age", "target": 42.0, "tolerance": 0.5},
      {"id": "share_high_income", "type": "share", "column": "income", "value": ">50K", "target_pct": 0.25, "tolerance_pct": 0.02},
      {"id": "max_hours", "type": "max", "column": "hours-per-week", "upper_bound": 60}
    ],
    "hard_rule_ids": ["eq_occupation_private", "max_hours"]
  },
  "statistical_metrics": {"enabled": true},
  "constraint_metrics": {"enabled": true},
  "coverage_metrics": {"enabled": true},
  "ml_efficacy_metrics": {"enabled": true, "estimators": {"classification": "xgboost", "regression": "random_forest"}},
  "plausibility_metrics": {"enabled": true},
  "privacy_metrics": {"enabled": true},
  "differential_privacy": {"enabled": false, "epsilon": null, "delta": null}
}
```
The evaluator validates configs (e.g., ensure rule columns exist, share targets sum ≤ 1). Missing metadata triggers inference with warnings.
If `--constraints-path` is provided, its rules override/augment the `constraints.rules` list after validation, allowing constraint management separate from other configs.

## 5. Architecture
```
Evaluator/
├── main.py              # CLI entry, argument parsing, logging setup
├── evaluator.py         # Orchestrates metric execution and aggregation
├── config.py            # Config schema loading/validation
├── data_loader.py       # Handles CSV loading, type casting, joins
├── metrics/
│   ├── __init__.py
│   ├── statistical.py   # Distribution alignment
│   ├── constraints.py   # Constraint satisfaction & support ratios
│   ├── coverage.py      # Higher-order diversity metrics
│   ├── ml_efficacy.py   # Downstream ML utility
│   ├── plausibility.py  # Wrapper around pretrained scorer
│   ├── privacy.py       # Proximity/privacy checks
│   └── dp.py            # Optional differential privacy adapter
├── reporting/
│   ├── writer.py        # JSON/CSV summaries
│   └── visualizer.py    # Optional plots (histograms, violation charts)
└── utils.py             # Shared helpers (seed setting, sampling, parallelism)
```
Each metrics submodule exposes a `compute(context: MetricContext) -> Dict` interface so new metrics can be plugged in without touching the orchestrator.

## 6. Metric Suites
### 6.1 Statistical Fidelity (metrics/statistical.py)
- **Inputs:** real & synthetic `DataFrame`, column metadata.
- **Metrics:** Wasserstein distance, KS statistic/p-value, Jensen-Shannon divergence, KL divergence, Mean/Absolute error of summary stats, correlation matrix norm, covariance shift, chi-square for categorical distributions.
- **Notes:** Support bootstrap CI for metrics to quantify uncertainty; handle mixed datatypes.

### 6.2 Constraint Adherence & Support (metrics/constraints.py)
- **Constraint Satisfaction Rate:** rows meeting each rule ÷ total rows; reported separately for hard vs soft rules.
- **Support Ratio Preservation:** compare observed share vs target share (`Δ%`).
- **Range/Boundary Compliance:** fraction within min/max bounds with violation counts.
- **Aggregate Consistency:** difference between requested aggregates (mean, sum) and observed values.
- **Conditional Rules:** support multi-column logic (e.g., `IF age < 18 THEN income = "None"`).
- **Outputs:** table with columns `[rule_id, type, target, observed, deviation, satisfied]` plus global pass/fail rate.

### 6.3 Coverage & Diversity (metrics/coverage.py)
- Pairwise/triwise contingency divergence (e.g., total variation distance across combos).
- Mutual information preservation for selected column sets.
- Unique ratio / duplication rate.
- Rare category retention and frequency rank shift.
- Missingness pattern alignment (if NA values are meaningful).

### 6.4 Machine Learning Efficacy (metrics/ml_efficacy.py)
- Train model(s) on synthetic data, evaluate on real holdout (and optionally the reverse for two-way transfer).
- Classification: accuracy, macro F1, AUROC/PR for binary tasks.
- Regression: RMSE, MAE, R².
- Report delta vs baseline model trained on real data only; include confidence intervals via repeated runs or cross-validation.

### 6.5 Plausibility (metrics/plausibility.py)
- Wrapper around external autoregressive scorer; requires `--model-path`.
- Metrics: average log-likelihood / perplexity, percentile thresholds for low-plausibility samples, outlier count.
- Provide option to annotate offending rows in output artifacts.

### 6.6 Privacy (metrics/privacy.py)
- Distance to Closest Record (DCR) distribution, min/avg distance, threshold breach rate.
- Nearest-neighbor overlap (exact match detection for categorical/hybrid).
- Optional membership or attribute inference attacks (simple classifiers) gated by configs.

### 6.7 Differential Privacy Placeholder (metrics/dp.py)
- If the underlying generator ships epsilon/delta, record them and validate they meet user thresholds.
- Otherwise emit `"status": "not_available"` with guidance to integrate training-time accounting.

## 7. Workflow
1. **Argument Parsing:** `main.py` loads CLI args, merges with configs, sets logging + random seeds.
2. **Config Validation:** ensure schema integrity, constraint sanity, metric compatibility (e.g., target column present for ML efficacy).
3. **Data Loading:** ingest synthetic (single or multi-file) and real CSVs, harmonize schemas, apply dtype coercion, align categorical domains, and optionally sample for scalability.
4. **Preprocessing:** handle missing values as configured, encode categoricals (one-hot or ordinal), standardize numerics when needed, materialize constraint masks.
5. **Metric Execution:** orchestrator passes a shared `MetricContext` to each enabled module; modules run independently and emit structured results plus warnings.
6. **Aggregation:** merge metric outputs, compute composite scores (e.g., weighted average) if requested, and attach metadata (run id, timestamps, git hash, versions).
7. **Reporting:** persist JSON summary, CSV tables (constraint results, metric-by-column tables), and optional plots; print concise CLI recap.
8. **Exit Codes:** non-zero exit when hard constraints fail or when critical modules error out (configurable).

## 8. Outputs & Reporting
- `summary.json`: all metrics, configuration snapshot, dataset stats, constraint verdicts.
- `constraints.csv`: per-rule diagnostics.
- `metrics/statistical.csv`, `metrics/coverage.csv`, etc., for drill-down.
- `plots/`: PNG/HTML visualizations if visualization module enabled.
- Console output: brief status per dimension plus pointers to detailed files.

## 9. Extensibility, Testing, and Quality
- **Plugin-Friendly:** metrics registered via entry points or config list, enabling future additions (e.g., fairness metrics).
- **Testing:** unit tests per metric module with synthetic fixtures; integration tests that run the CLI on sample datasets with known scores; regression tests for constraint evaluation logic.
- **Performance:** optional parallel execution per metric category; streaming readers for large CSVs; ability to down-sample for heavy metrics while keeping full data for constraint checks.
- **Documentation:** README sections for metric definitions, constraint DSL, troubleshooting, and examples.
