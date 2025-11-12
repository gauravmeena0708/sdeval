# Synthetic Data Evaluator Plan

## 1. Goals & Scope

### Primary Goal
Deliver a CLI tool that evaluates synthetic tabular datasets (both **conditional** and **unconditional**) against real training data by measuring quality across multiple dimensions: statistical fidelity, constraint adherence, coverage/diversity, ML utility, plausibility, and privacy.

### Generator Types Supported

**1. Unconditional Generators**
- Generate data by learning the joint distribution of all features without explicit constraints
- Examples: GANs, VAEs, diffusion models, copula-based methods
- Evaluation focus: How well does the learned distribution naturally satisfy domain requirements?

**2. Conditional/Constraint-Aware Generators**
- Trained or configured to satisfy specific business rules or statistical targets during generation
- Examples: Constrained optimization (e.g., linear programming), rule-based synthesis, conditional GANs with constraint enforcement
- Evaluation focus: Does the generator successfully enforce its declared constraints while maintaining quality?

### Constraint Role in Evaluation

User-supplied constraints serve as **evaluation criteria**, with different interpretations depending on generator type:

**For Unconditional Generators:**
- Constraints are **post-hoc quality checks**: Does the synthetic data happen to satisfy domain requirements?
- Violations indicate the generator hasn't learned these patterns from the real data
- Example: If real data has `mean(age) = 42`, does synthetic data achieve similar statistics naturally?

**For Conditional Generators:**
- Constraints are **verification of promised guarantees**: Does the generator enforce what it claims?
- Violations indicate implementation bugs or insufficient constraint enforcement
- Example: If generator claims to enforce `share(gender=F) = 0.52`, verify this holds in output

### Constraint Types
Constraints represent:
- **Domain knowledge**: Business rules that must hold (e.g., "age >= 18 for employed workers")
- **Statistical targets**: Desired distributional properties (e.g., "maintain 50/50 gender ratio")
- **Quality thresholds**: Acceptable bounds for synthetic data (e.g., "max hours worked <= 80")
- **Generator specifications**: Declared guarantees from conditional generators (e.g., "equality constraint on status column")

### Scope
- Support CSV inputs (single file or directory) for batch evaluation
- Produce machine-readable reports (JSON) plus human-friendly summaries
- Enable comparison of multiple synthetic generators (conditional vs unconditional, or different architectures)
- Provide actionable feedback: which constraints are violated, by how much, and where quality issues exist
- Flexible constraint interpretation: users specify constraints based on their evaluation goals (domain requirements, generator verification, or both)

### Non-Goals
- Building or training synthetic data generators
- Enforcing constraints during generation (generator is treated as black-box)
- Automatically inferring which generators are conditional vs unconditional (user context-dependent)
- Guaranteeing formal differential privacy (placeholder only for future integration)
- Supporting unstructured data (images, text, time series)

## 2. Success Criteria & Maturity Levels

This project follows a phased implementation approach:

### Phase 1: Core Functionality (MVP) ✅
- **Constraint Satisfaction:** Report rate and deviation for every declared rule (equality, mean, share, min/max); flag hard violations separately from soft targets.
- **Basic Fidelity:** 1D distribution alignment (Wasserstein, mean/std comparison), alpha/beta precision for categorical coverage.
- **ML Utility:** Train-on-synthetic/test-on-real workflow with RandomForest models (classification + regression).
- **Basic Privacy:** k-NN distance metrics (DCR, NNDR) for proximity-based leakage checks.
- **Basic Coverage:** Uniqueness ratio, rare category retention, missingness alignment.
- **Plausibility:** Integration with external autoregressive model scorer (requires separate training).

### Phase 2: Production Readiness (Current Target) 🎯
- **Reproducibility:** Deterministic runs via global seed propagation (numpy, sklearn, torch).
- **Config Validation:** Pre-flight checks for column existence, constraint sanity (share targets ≤ 1), metric compatibility.
- **Testing:** Unit tests per metric module with synthetic fixtures; integration tests with known-good outputs.
- **Exit Codes:** Non-zero exit when hard constraints fail or critical errors occur.
- **Structured Reporting:** Per-metric CSV tables + consolidated JSON summary.
- **Documentation:** README with quickstart, metric definitions, constraint DSL reference, troubleshooting.

### Phase 3: Advanced Features (Roadmap) 🔮
- **Enhanced Fidelity:** KS test, JS/KL divergence, correlation matrix norms, chi-square for categorical distributions, bootstrap confidence intervals.
- **Advanced Coverage:** Pairwise/triwise contingency analysis, mutual information preservation.
- **Advanced Privacy:** DCR distribution analysis, configurable thresholds, membership inference attacks.
- **Conditional Constraints:** Multi-column logic (e.g., `IF age < 18 THEN income = "None"`).
- **Visualization:** Histograms, QQ plots, constraint violation charts.
- **Performance:** Parallel metric execution, streaming readers for large CSVs, smart sampling.

## 3. CLI Surface
```bash
python -m sdeval.main \
    --input-path <synthetic_csv_or_folder> \
    --training-data-csv-path <real_data_csv> \
    --output-dir <output_dir> \
    [--model-path <plausibility_model_dir>] \
    [--configs <configs_json>] \
    [--constraints-path <constraints_json>] \
    [--metrics <metric1> <metric2> ...] \
    [--overwrite] \
    [--seed <int>]
```
- `--input-path`: synthetic data source; directory iteration is recursive (CSV only).
- `--training-data-csv-path`: required reference data.
- `--output-dir`: results root; outputs written directly unless subdirectory structure created.
- `--model-path`: optional path to pre-trained plausibility model (triggers auto-training if missing and enabled in config).
- `--configs`: JSON describing schema, target column, constraint rules, metric toggles, etc.
- `--constraints-path`: optional standalone JSON defining constraint rules when not bundled inside `--configs`.
- `--metrics`: optional whitelist of metrics to run (default: all enabled in config or ["statistical", "constraints", "coverage", "ml_efficacy", "plausibility", "privacy", "dp"]).
- `--overwrite`: overwrite existing output directory without prompting.
- `--seed`: random seed for reproducible sampling/model training (Phase 2 target: full propagation to all RNG sources).

## 4. Configuration Contract (`configs.json`)

### 4.1 Schema Definition
Key fields (all optional unless noted):
```json
{
  "target_column": "income",
  "task_type": "classification",
  "categorical_columns": ["workclass", "education"],
  "numerical_columns": ["age", "fnlwgt", "hours-per-week"],
  "constraints": {
    "rules": [
      {"id": "eq_occupation_private", "type": "equality", "column": "occupation", "value": "Private"},
      {"id": "mean_age", "type": "mean", "column": "age", "target": 42.0, "tolerance": 0.5},
      {"id": "share_high_income", "type": "share", "column": "income", "value": ">50K", "target_pct": 0.25, "tolerance_pct": 0.02},
      {"id": "max_hours", "type": "max", "column": "hours-per-week", "upper_bound": 60}
    ]
  },
  "plausibility_metrics": {
    "enabled": true,
    "train_if_missing": false,
    "train_epochs": 50
  },
  "privacy_metrics": {
    "enabled": true,
    "dcr_threshold": 1e-8
  },
  "seed": 42
}
```

### 4.2 Constraint Rule Types

**Implemented (Phase 1):**
- `equality`: Column must exactly match a value (e.g., `{"type": "equality", "column": "status", "value": "Active"}`)
- `expression`: Generic expression evaluated via pandas.eval (e.g., `{"type": "expression", "expression": "age >= 18"}`)
- `share`: Population share target (e.g., `{"type": "share", "column": "gender", "value": "F", "target_pct": 0.52, "tolerance_pct": 0.02}`)
- `mean`: Mean value target (e.g., `{"type": "mean", "column": "salary", "target": 75000, "tolerance": 5000}`)
- `min`/`max`: Boundary constraints (e.g., `{"type": "max", "column": "age", "upper_bound": 100}`)

All rules support:
- `hard`: boolean (default: true for equality/expression/min/max, false for mean/share) - determines if violation causes exit code failure
- `tolerance` or `tolerance_pct`: allowed deviation for soft constraints

**Typical Usage Patterns:**

*Evaluating Unconditional Generators:*
```json
{
  "constraints": {
    "rules": [
      {"id": "mean_age", "type": "mean", "column": "age", "target": 42.0, "tolerance": 2.0, "hard": false},
      {"id": "share_female", "type": "share", "column": "gender", "value": "F", "target_pct": 0.52, "tolerance_pct": 0.05, "hard": false}
    ]
  }
}
```
- Use soft constraints (hard=false) with generous tolerances
- Focus on statistical similarity, not exact enforcement
- Violations indicate the generator didn't learn the distribution well

*Evaluating Conditional Generators:*
```json
{
  "constraints": {
    "rules": [
      {"id": "enforce_adult", "type": "expression", "expression": "age >= 18", "hard": true},
      {"id": "enforce_share", "type": "share", "column": "income", "value": ">50K", "target_pct": 0.25, "tolerance_pct": 0.01, "hard": true}
    ]
  }
}
```
- Use hard constraints (hard=true) with tight tolerances
- Reflects the generator's declared guarantees
- Violations indicate bugs or insufficient constraint enforcement

**Phase 3 Roadmap:**
- Conditional rules: `{"type": "conditional", "condition": "age < 18", "then": {"column": "income", "value": null}}`
- Multi-column expressions: complex pandas.eval expressions with proper parsing

### 4.3 Config Validation (Phase 2 Target)

Validation checks to implement:
1. **Column Existence**: All constraint columns exist in actual DataFrames
2. **Constraint Sanity**: Sum of share targets ≤ 1.0 per column
3. **Metric Compatibility**: If `ml_efficacy` enabled, ensure `target_column` is set
4. **Type Consistency**: Categorical columns aren't used in mean/min/max constraints
5. **Expression Syntax**: pandas.eval expressions parse without errors

Current State: Only basic file existence checks implemented.

### 4.4 Constraint Merging
If `--constraints-path` is provided, its rules are merged with `constraints.rules` from the main config. Duplicate rule IDs (matched by `id` field) in `--constraints-path` override those in the main config.

## 5. Architecture

### 5.1 Module Structure (Current)
```
sdeval/
├── __init__.py
├── main.py              # CLI entry, argument parsing
├── evaluator.py         # Orchestrates metric execution
├── config.py            # Config loading (basic file checks only)
├── data_loader.py       # CSV loading (no preprocessing yet)
├── metrics/
│   ├── __init__.py      # MetricContext + registry pattern
│   ├── statistical.py   # ✅ Alpha/beta, mean/std, Wasserstein
│   ├── constraints.py   # ✅ Equality, mean, share, min/max rules
│   ├── coverage.py      # ✅ Uniqueness, rare categories, missingness
│   ├── ml_efficacy.py   # ✅ RandomForest train-on-syn/test-on-real
│   ├── plausibility.py  # ✅ Autoregressive scorer integration (complex)
│   ├── privacy.py       # ✅ k-NN DCR, NNDR
│   └── dp.py            # ⚠️  Placeholder stub
└── reporting/
    ├── __init__.py
    └── writer.py        # ✅ Single JSON summary output
```

### 5.2 MetricContext Design
Current implementation (`metrics/__init__.py:13-19`):
```python
@dataclass
class MetricContext:
    real_df: pd.DataFrame
    synthetic_df: pd.DataFrame
    settings: EvaluatorSettings
    synthetic_path: str
```

**Phase 2 Enhancement Plan:**
Add pre-computed metadata to reduce duplicate logic across metrics:
- `categorical_columns: List[str]` (auto-detected)
- `numerical_columns: List[str]` (auto-detected)
- `column_metadata: Dict[str, ColumnMeta]` (dtype, nunique, null_pct, etc.)
- `logger: logging.Logger` (for consistent logging)

### 5.3 Module Status & Roadmap

| Module | Phase 1 Status | Phase 2 Needs | Phase 3 Additions |
|--------|----------------|---------------|-------------------|
| **utils.py** | ❌ Missing | Seed propagation, column detection helpers | Sampling, parallelism |
| **config.py** | ⚠️ Basic | Schema validation, constraint sanity checks | Advanced expression parsing |
| **data_loader.py** | ⚠️ Basic | Preprocessing, schema harmonization | Streaming, smart sampling |
| **statistical.py** | ⚠️ Minimal | KS test, chi-square | Correlation, KL/JS, bootstrap CIs |
| **coverage.py** | ⚠️ Basic | n/a | Pairwise/triwise, mutual information |
| **privacy.py** | ⚠️ Basic | Configurable thresholds | DCR distribution, membership attacks |
| **plausibility.py** | ✅ Complete | Document complexity | Outlier annotation |
| **reporting/writer.py** | ⚠️ Minimal | Per-metric CSVs | Composite scores, metadata |
| **reporting/visualizer.py** | ❌ Missing | n/a | Histograms, QQ plots, violation charts |
| **tests/** | ❌ Missing | Full unit + integration coverage | Performance benchmarks |

Each metric module follows the registry pattern:
```python
@register_metric("metric_name")
def compute_metric_name(ctx: MetricContext) -> Dict[str, Any]:
    # Returns flat dict of metrics + metadata
```

## 6. Metric Suites

### 6.1 Statistical Fidelity (metrics/statistical.py)

**Phase 1 Implemented:**
- `alpha_precision`: Fraction of synthetic categorical values present in real data
- `beta_recall`: Fraction of real categorical values covered by synthetic data
- `mean_abs_mean_diff`: Average absolute difference in column means
- `mean_abs_std_diff`: Average absolute difference in column standard deviations
- `avg_wasserstein`: Average Wasserstein distance for numerical columns (requires scipy)

**Phase 2 Priority Additions:**
- KS statistic + p-value for numerical columns
- Chi-square test for categorical distributions

**Phase 3 Roadmap:**
- Jensen-Shannon & KL divergence
- Correlation matrix Frobenius norm
- Covariance shift detection
- Bootstrap confidence intervals for all metrics

### 6.2 Constraint Adherence (metrics/constraints.py)

**Phase 1 Implemented:**
- Equality/expression rules via pandas.eval
- Share targets with tolerance (e.g., 52% female ± 2%)
- Mean targets with tolerance (e.g., avg_age = 42 ± 0.5)
- Min/max boundary constraints
- Hard vs soft rule classification
- Per-rule pass/fail with deviation metrics

**Output Structure:**
```json
{
  "constraints_enabled": true,
  "constraints_total_rules": 4,
  "constraints_passed_rules": 3,
  "constraints_hard_failures": 1,
  "constraint_rule_details": [
    {
      "rule_id": "mean_age",
      "type": "mean",
      "observed_value": 42.3,
      "target_value": 42.0,
      "deviation": 0.3,
      "tolerance": 0.5,
      "hard": false,
      "passed": true
    }
  ]
}
```

**Phase 3 Roadmap:**
- Conditional rules: `IF condition THEN constraint`
- Sum aggregation constraints
- Advanced expression parser (replace regex with proper AST)

### 6.3 Coverage & Diversity (metrics/coverage.py)

**Phase 1 Implemented:**
- `coverage_unique_ratio`: Fraction of unique rows in synthetic data
- `coverage_rare_category_retention`: Fraction of rare real categories present in synthetic
- `coverage_missing_category_ratio`: Fraction of real categories missing in synthetic
- `coverage_missingness_delta`: Difference in overall null rates

**Phase 3 Roadmap:**
- Pairwise/triwise contingency total variation distance
- Mutual information preservation
- Frequency rank correlation (Spearman's rho)

### 6.4 Machine Learning Efficacy (metrics/ml_efficacy.py)

**Phase 1 Implemented:**
- Train RandomForest on synthetic, evaluate on real (TSTR)
- Auto-detection of task type (classification vs regression)
- Classification: `ml_accuracy`, `ml_f1_macro`
- Regression: `ml_rmse`, `ml_mae`, `ml_r2`
- Automatic feature alignment between datasets

**Phase 2 Additions:**
- Baseline comparison (train on real, test on real) to compute utility delta
- Seed-controlled reproducibility

**Phase 3 Roadmap:**
- Two-way transfer: TRTS (train-on-real/test-on-synthetic)
- Support for XGBoost and other estimators
- Cross-validation for confidence intervals
- AUROC/AUPRC for binary classification

### 6.5 Plausibility (metrics/plausibility.py)

**Phase 1 Implemented:**
- Integration with external autoregressive model (submodules/plausibility)
- Auto-training: if model missing, runs `plausibility/tp.py` subprocess
- Auto-binning: generates `bin_info.json` for numerical columns
- Caching: stores results in `expdir/output_{dataset}/plausibility.csv`
- `plausibility_avg`: average per-row negative log-likelihood

**Implementation Notes:**
This module is more complex than originally planned (250 lines vs simple wrapper). It handles:
- Dataset name inference from training path
- Bin info generation (default 10 bins per numerical column)
- Categorical encoding via learned mappings
- Model architecture inference (vocab_size, d_model)
- Batch inference with torch

**Phase 2 Needs:**
- Documentation of auto-training behavior and caching logic
- Clearer error messages when plausibility module fails to import

**Phase 3 Roadmap:**
- Percentile-based outlier detection
- Row-level annotation in output files

### 6.6 Privacy (metrics/privacy.py)

**Phase 1 Implemented:**
- `privacy_dcr`: Distance to Closest Record rate (threshold: 1e-8)
- `privacy_nndr`: Nearest Neighbor Distance Ratio (mean)
- `privacy_knn_distance`: Mean distance to nearest real neighbor

**Phase 2 Priority:**
- Configurable DCR threshold via config
- Min/max distance statistics

**Phase 3 Roadmap:**
- DCR distribution (histogram)
- Exact match detection for categorical columns
- Membership inference attack (train classifier to detect synthetic vs real)
- Attribute inference attack

### 6.7 Differential Privacy (metrics/dp.py)

**Current Status:** Stub/placeholder only

**Phase 3 Implementation Plan:**
- If config specifies `{"differential_privacy": {"enabled": true, "epsilon": 1.0, "delta": 1e-5}}`
- Check if generator metadata includes ε/δ (e.g., from DP-SGD training)
- Validate privacy budget meets requirements
- Otherwise emit: `{"dp_status": "not_available", "message": "No DP guarantees detected"}`

## 7. Execution Workflow

### Phase 1 (Current Implementation)
1. **Argument Parsing** (`main.py:25-36`): Load CLI args, merge with config JSON
2. **Config Loading** (`config.py:38-87`): Basic file existence validation only
3. **Data Loading** (`data_loader.py:10-28`): Simple `pd.read_csv()` for real + synthetic files
4. **Metric Execution** (`evaluator.py:16-44`): Sequential loop through enabled metrics
5. **Reporting** (`reporting/writer.py:8-13`): Single JSON file per synthetic input
6. **Exit** (`main.py:32-36`): Always returns 0 on success, 1 on exception

### Phase 2 Enhancements
1. **Seed Propagation**: Set `numpy.random.seed()`, `random.seed()`, `torch.manual_seed()` early in `main.py`
2. **Config Validation**: Pre-flight checks before data loading (see section 4.3)
3. **Schema Harmonization**: Ensure real/synthetic have compatible columns, align categorical domains
4. **Exit Codes**: Return 2 when hard constraints fail, 1 for errors, 0 for success
5. **Structured Reporting**:
   - `{output_dir}/summary.json` (all metrics + metadata)
   - `{output_dir}/constraints.csv` (per-rule table)
   - `{output_dir}/metrics/{metric_name}.csv` (per-metric drill-down)

### Phase 3 Additions
- **Preprocessing Pipeline**: Missing value imputation, categorical encoding, numeric standardization
- **Parallel Execution**: `concurrent.futures` for independent metrics
- **Metadata Tracking**: Git hash, package versions, dataset stats in summary
- **Composite Scores**: Weighted average across metric dimensions

## 8. Outputs & Reporting

### Current (Phase 1)
```
output_dir/
└── {synthetic_filename}_summary.json
```

Single JSON file containing all metrics as flat dictionary.

### Phase 2 Target
```
output_dir/
├── summary.json                    # Consolidated report
├── constraints.csv                 # Per-rule breakdown
├── metrics/
│   ├── statistical.csv             # Per-column statistical metrics
│   ├── coverage.csv                # Per-column coverage metrics
│   └── privacy.csv                 # Privacy metrics detail
└── metadata.json                   # Run info, versions, timestamps
```

### Phase 3 Vision
```
output_dir/
├── summary.json
├── constraints.csv
├── metrics/*.csv
├── metadata.json
└── plots/
    ├── distributions/              # Real vs synthetic histograms
    │   ├── age.png
    │   └── income.png
    ├── constraints_violations.png  # Bar chart of rule pass/fail
    └── correlation_heatmap.png     # Real vs synthetic correlation matrices
```

**Console Output (Phase 2):**
```
[INFO] Evaluating synthetic/sample1.csv against datasets/adult/train.csv
[INFO] Metrics: statistical, constraints, ml_efficacy, privacy

✓ Statistical Fidelity:  alpha=0.95, beta=0.92, wasserstein=0.034
✓ Constraints:           3/4 passed (1 hard failure: max_hours)
✓ ML Efficacy:           accuracy=0.84 (TSTR)
✓ Privacy:               DCR=0.02, mean_distance=2.4

[ERROR] Hard constraint failure: max_hours
[INFO] Detailed results: outputs/sample1_summary.json
Exit code: 2
```

## 9. Testing, Quality & Extensibility

### Phase 2 Testing Priorities

**Unit Tests (per module):**
- `tests/test_statistical.py`: Verify Wasserstein, mean/std calculations with synthetic fixtures
- `tests/test_constraints.py`: Test all rule types (equality, mean, share, min/max, expression)
- `tests/test_coverage.py`: Uniqueness, rare category detection
- `tests/test_ml_efficacy.py`: Train/test split logic, feature alignment
- `tests/test_privacy.py`: k-NN distance calculations
- `tests/test_config.py`: Config loading, merging, validation

**Integration Tests:**
- End-to-end CLI run with known-good dataset
- Verify output file structure
- Regression tests for constraint evaluation (golden outputs)

**Test Fixtures:**
- Small synthetic datasets (50-100 rows) with known properties
- Configs with intentional violations
- Edge cases: empty DataFrames, single-column data, all-null columns

### Testing Commands
```bash
# Run all tests
pytest tests/ -v

# Coverage report
pytest tests/ --cov=sdeval --cov-report=html

# Integration test
python -m sdeval.main --input-path tests/fixtures/synthetic.csv \
                      --training-data-csv-path tests/fixtures/real.csv \
                      --output-dir /tmp/test_output \
                      --seed 42
```

### Extensibility (Plugin Pattern)

The registry pattern (`metrics/__init__.py:27-41`) enables easy metric additions:

```python
# New metric in sdeval/metrics/fairness.py
from . import register_metric, MetricContext

@register_metric("fairness")
def compute_fairness_metrics(ctx: MetricContext) -> Dict[str, Any]:
    # Implement demographic parity, equalized odds, etc.
    return {"fairness_dp": 0.95, "fairness_eo": 0.92}
```

No changes to `evaluator.py` needed - just import the module in `metrics/__init__.py`.

### Performance Targets (Phase 3)

- **Small datasets** (< 10k rows): < 30 seconds end-to-end
- **Medium datasets** (10k-100k rows): < 5 minutes
- **Large datasets** (> 100k rows): Streaming mode with optional sampling

**Optimization Strategies:**
- Parallel metric execution (each metric is independent)
- Smart sampling: full data for constraints, 10k sample for expensive metrics
- Caching: Plausibility already caches results per file
- Incremental evaluation: Skip unchanged files in directory mode

### Documentation Roadmap

**Phase 2:**
- README.md with quickstart, installation, basic examples
- Constraint DSL reference
- Troubleshooting guide (common errors)

**Phase 3:**
- Full metric definitions (mathematical formulas)
- Interpretation guide (what do the metrics mean?)
- Advanced use cases (CI/CD integration, batch evaluation)
- Contributing guide for new metrics

---

## Summary & Implementation Status

### What Works Today (Phase 1 ✅)

The tool is a **functional MVP** suitable for:
- Evaluating both conditional and unconditional synthetic data generators
- Checking constraint adherence (equality, mean, share, boundary rules)
- Measuring basic statistical fidelity (Wasserstein, mean/std alignment, categorical coverage)
- Assessing ML utility via train-on-synthetic/test-on-real workflow
- Detecting privacy leakage via k-NN distance metrics
- Scoring plausibility with external autoregressive models

**Use Cases:**
- Compare unconditional generators (GAN, VAE, etc.) by how well they learn distributional properties
- Verify conditional generators enforce their declared constraints
- Benchmark multiple generators on the same dataset with standardized metrics

**Quick Start:**
```bash
python -m sdeval.main \
    --input-path synthetic/adult_synthetic.csv \
    --training-data-csv-path datasets/adult/train.csv \
    --output-dir outputs/ \
    --seed 42
```

### Critical Gaps (Phase 2 Targets 🎯)

Before production use, implement:
1. **Testing** - No tests exist; add unit + integration coverage
2. **Config Validation** - Pre-flight column/constraint sanity checks
3. **Reproducibility** - Global seed propagation (numpy, torch, sklearn)
4. **Exit Codes** - Non-zero exit for hard constraint failures
5. **Documentation** - README with examples, constraint DSL reference

**Estimated Effort:** 2-3 weeks for one developer

### Future Enhancements (Phase 3 🔮)

Advanced features for research/production scale:
- Enhanced statistical tests (KS, chi-square, correlation analysis)
- Visualization module (distribution plots, constraint violation charts)
- Parallel metric execution for performance
- Advanced privacy metrics (membership inference)
- Conditional constraints and complex expression parsing

**Estimated Effort:** 1-2 months for advanced feature set

### Key Design Decisions

1. **Registry Pattern:** Metrics self-register via decorators - easy to extend
2. **No Preprocessing:** Raw DataFrames passed to metrics - each module handles its own prep (leads to duplicate logic)
3. **Plausibility Complexity:** Auto-training and caching add 200+ lines beyond original "simple wrapper" plan
4. **Single JSON Output:** Flat dictionary in one file (Phase 2 will add structured CSVs)
5. **Sequential Execution:** No parallelism yet (acceptable for MVP)

### Recommendations

**For evaluating unconditional generators:**
- Define constraints based on real data properties (use real data stats as targets)
- Use soft constraints (hard=false) with reasonable tolerances (e.g., ±5% for shares, ±10% for means)
- Focus on statistical metrics (fidelity, coverage) rather than exact constraint enforcement
- Compare multiple generators using the same constraint set for fair benchmarking

**For evaluating conditional generators:**
- Define constraints matching the generator's declared guarantees
- Use hard constraints (hard=true) with tight tolerances (e.g., ±1% for enforced shares)
- Prioritize constraint metrics to verify enforcement correctness
- Set exit codes to fail on hard constraint violations for CI/CD integration

**For immediate use (both types):**
- Manually verify outputs against expected properties (exit code enforcement not yet implemented)
- Use fixed seeds for reproducibility where possible
- Start with a subset of metrics to understand tool behavior

**For production deployment:**
- Complete Phase 2 checklist above
- Add monitoring/logging infrastructure
- Create golden test outputs for regression testing
- Document plausibility module's auto-training behavior
- Establish clear thresholds for constraint pass/fail based on generator type

**For research/advanced use:**
- Contribute Phase 3 features as needed
- Consider forking for specialized metrics (fairness, causality, etc.)
- Extend MetricContext with pre-computed metadata to reduce duplication
- Add generator metadata field to track conditional vs unconditional type
