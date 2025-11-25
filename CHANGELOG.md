# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-01-13

### Added - CLI Enhancements
- **Progress bars** with tqdm for long-running evaluations
- **Verbose mode** (`--verbose`/`-v`) with detailed progress logging
- **Quiet mode** (`--quiet`/`-q`) for minimal output
- **Timestamped output directories** (`--timestamp`/`-t`) for organized runs
- **Batch evaluation mode** with automatic directory processing
- **Comparison table generation** (`--compare`/`-c`) for batch evaluations
  - Automatically creates CSV with flattened metrics across all files
- **Enhanced help text** with example commands
- **Improved error messages** with optional stack traces in verbose mode
- **Progress indicators** for both file processing and metric computation

### Changed - CLI
- CLI now shows beautiful header with ASCII art
- Better structured output with sections
- Summary statistics at the end of evaluation
- Per-metric status reporting in verbose mode

### Dependencies
- Added `tqdm` for progress bars

## [1.1.0] - 2025-01-12

### Added - Nomenclature Standardization
- **ML Efficacy metrics** (`ml_efficacy.py`)
  - Train-on-Synthetic, Test-on-Real (TSTR) evaluation
  - Support for classification (F1, Accuracy) and regression (MAE, MSE, R2)
  - Auto-detection of task type
  - Requires `target_column` in configuration
- **Constraint Satisfaction metrics** (`constraints.py`)
  - Simple categorical constraints (e.g., `education=Bachelors`)
  - Multiple constraints with AND logic
  - Complex constraints via pandas eval expressions
  - Statistical bounds (mean, min, max)
  - Population share targets
  - Functions: `compute_constraint_satisfaction_rate()` and `compute_constraint_support()`
- Example script: `example_constraint_evaluation.py`

### Changed - Breaking Changes
- **Standardized metric output keys** across all modules
  - All metrics now use `{category}_{metric_name}` pattern
  - Statistical: `alpha_precision` → `statistical_alpha_precision`
  - Coverage: `uniqueness_ratio` → `coverage_uniqueness_ratio`
  - ML Efficacy: All keys prefixed with `ml_efficacy_`
  - Constraints: All keys prefixed with `constraints_`
- **Renamed configuration parameters**
  - `training_data_path` → `real_data_path`
  - `--training-data-csv-path` → `--real-data-csv-path`
- **Fixed single-letter variables**
  - `privacy.py`: `R`, `G` → `real_data_array`, `synthetic_data_array`
- **Standardized DataFrame parameter naming**
  - All functions use `real_df` and `synthetic_df` consistently
  - No more generic `df` parameters
- **Standardized column variable naming**
  - Loop variables: `c` → `col`
  - Function parameters: `column` for full names
- **Fixed function pluralization**
  - `compute_constraint_metrics` → `compute_constraints_metrics`
- **Standardized status patterns**
  - All metrics use `{metric}_reason` (not `{metric}_status`)
  - `dp.py`: `dp_status` → `dp_reason`

### Fixed
- Inconsistent terminology across codebase
- Mixed abbreviation usage
- Variable naming inconsistencies

### Updated
- All tests (80 passing) to use new nomenclature
- README.md with standardized examples
- `example_evaluation.py` with new metric keys
- All docstrings and comments

## [1.0.0] - 2024-11-12

### Added - Initial Release
- **Statistical Fidelity metrics** (5 metrics)
  - Alpha Precision
  - Beta Recall
  - Mean Absolute Difference
  - Std Absolute Difference
  - Wasserstein Distance
- **Coverage & Diversity metrics** (4 metrics)
  - Uniqueness Ratio
  - Rare Category Retention
  - Missing Category Ratio
  - Missingness Delta
- **Privacy metrics** (3 metrics)
  - DCR (Distance to Closest Record)
  - NNDR (Nearest Neighbor Distance Ratio)
  - Mean k-NN Distance
- **Core infrastructure**
  - Automatic column type detection
  - CSV data loading
  - Distribution visualization
  - JSON output generation
  - Metric registry system
- **Quality assurance**
  - 61 comprehensive tests
  - GitHub Actions CI/CD
  - Cross-platform support
- **Documentation**
  - Complete README
  - API documentation
  - Working demo script
  - Metric explanations

---

## Migration Guide

### Upgrading from 1.0.0 to 1.1.0+

#### Update Metric Key References

**Before (1.0.0):**
```python
stats = compute_statistical_metrics(real_df, synthetic_df, num_cols, cat_cols)
print(stats['alpha_precision'])  # Old key
print(stats['beta_recall'])      # Old key
```

**After (1.1.0+):**
```python
stats = compute_statistical_metrics(real_df, synthetic_df, num_cols, cat_cols)
print(stats['statistical_alpha_precision'])  # New key
print(stats['statistical_beta_recall'])      # New key
```

#### Update CLI Arguments

**Before (1.0.0):**
```bash
python -m sdeval.main --training-data-csv-path real.csv --input-path synthetic.csv --output-dir outputs
```

**After (1.1.0+):**
```bash
python -m sdeval.main --real-data-csv-path real.csv --input-path synthetic.csv --output-dir outputs
```

#### Update Configuration Files

**Before (1.0.0):**
```json
{
  "training_data_csv_path": "real.csv",
  "input_path": "synthetic.csv"
}
```

**After (1.1.0+):**
```json
{
  "real_data_csv_path": "real.csv",
  "input_path": "synthetic.csv"
}
```

#### Complete Key Mapping

| Old Key (1.0.0) | New Key (1.1.0+) |
|-----------------|------------------|
| `alpha_precision` | `statistical_alpha_precision` |
| `beta_recall` | `statistical_beta_recall` |
| `mean_abs_mean_diff` | `statistical_mean_abs_mean_diff` |
| `mean_abs_std_diff` | `statistical_mean_abs_std_diff` |
| `avg_wasserstein` | `statistical_avg_wasserstein` |
| `uniqueness_ratio` | `coverage_uniqueness_ratio` |
| `rare_category_retention` | `coverage_rare_category_retention` |
| `missing_category_ratio` | `coverage_missing_category_ratio` |
| `missingness_delta` | `coverage_missingness_delta` |
| `dcr_rate` | `privacy_dcr` (in registry) |
| `nndr_mean` | `privacy_nndr` (in registry) |
| `mean_knn_distance` | `privacy_knn_distance` (in registry) |

**Note:** Public API functions (e.g., `compute_dcr()`, `compute_nndr()`) return unprefixed keys. Only registry functions return prefixed keys.

---

## Links

- [GitHub Repository](https://github.com/gauravmeena0708/sdeval)
- [Issue Tracker](https://github.com/gauravmeena0708/sdeval/issues)
- [Documentation](https://github.com/gauravmeena0708/sdeval/blob/main/README.md)
