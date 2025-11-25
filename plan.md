# Synthetic Data Evaluator - Current Status & Future Roadmap

## 🎯 Current Status

**Version:** 1.3.0 (Visualization & Bulk Automation)
**Date:** January 16, 2025
**Status:** ✅ Production Ready with Bulk tooling & diagnostics

### What's Working

✅ **Core Metrics Implemented (12+ total)**
- Statistical Fidelity: 5 metrics
- Coverage & Diversity: 4 metrics
- Privacy Analysis: 3 metrics
- **ML Efficacy: TSTR evaluation** (NEW!)
- **ML Efficacy: TSTR evaluation** (NEW!)
- **Constraint Satisfaction: Categorical constraints** (NEW!)
- **Differential Privacy Tools: Mechanisms & Accounting** (NEW!)

✅ **Infrastructure**
- Automatic column type detection
- CSV data loading
- JSON output generation
- **Distribution, QQ, correlation, constraint plots via `--visualize`** (NEW!)
- **Progress bars with tqdm**
- **Timestamped output directories**
- **Batch evaluation with comparison tables**

✅ **Enhanced CLI & Bulk Runner** (UPDATED!)
- Verbose/quiet modes
- Progress indicators
- Batch processing (`python -m sdeval.main`, `evaluate_bulk.py`)
- Comparison table generation
- Improved error messages
- Example commands in help
- **Optional visualization flag (`--visualize`)**
- **Radar dashboards (data-quality & privacy) + Excel summaries in bulk mode** (NEW!)

✅ **Quality Assurance**
- 80 comprehensive tests (updated from 61)
- GitHub Actions CI/CD
- Cross-platform support (Windows, macOS, Linux)
- Python 3.9+ support
- **Standardized nomenclature across codebase** (NEW!)

✅ **Documentation**
- Complete README with examples
- Working demo scripts
- API documentation
- Metric explanations
- **Updated Quick Start for CLI/bulk visualization** (NEW!)

### Quick Links

- **Completed Tasks:** See [COMPLETED_TASKS.md](./COMPLETED_TASKS.md)
- **Usage Guide:** See [README.md](./README.md)
- **Demo:** Run `python example_evaluation.py`
- **Tests:** Run `pytest tests/`

---

## 🔮 Future Enhancements (Optional)

These are potential improvements for future versions. Current version is production-ready without these.

### Phase 8: Enhanced Statistical Metrics (Future)

- [x] Add KS (Kolmogorov-Smirnov) test
  - [x] Per-column KS statistic
  - [x] P-values for distribution similarity (aggregate average available)
- [x] Add Chi-square test for categorical distributions
  - [x] Test for independence / goodness-of-fit
  - [x] Expected vs observed frequencies
- [x] Add Jensen-Shannon divergence
  - [x] Alternative to Wasserstein for distributions
- [x] Add correlation analysis
  - [x] Correlation matrix comparison
  - [x] Frobenius norm of difference
- [ ] Add bootstrap confidence intervals
  - [ ] For all statistical metrics
  - [ ] Uncertainty quantification

**Priority:** Medium
**Estimated Effort:** 2-3 weeks

---

### Phase 9: Advanced Coverage Metrics (Future)

- [ ] Pairwise contingency analysis
  - [ ] Two-way frequency tables
  - [ ] Chi-square for pairs
- [ ] Triwise contingency analysis
  - [ ] Three-way frequency tables
- [ ] Mutual information preservation
  - [ ] Information-theoretic measure
  - [ ] Feature dependency preservation
- [ ] Frequency rank correlation
  - [ ] Spearman's rho for category frequencies

**Priority:** Low
**Estimated Effort:** 1-2 weeks

---

- ### Phase 10: Advanced Privacy Metrics (Future)
 
 - [x] k-anonymity support
   - [x] Requires quasi-identifier configuration
   - [x] Equivalence class analysis
   - [ ] See [k-anonymity design doc](docs/k-anonymity-design.md) (to be created)
 - [x] DCR distribution analysis
   - [x] Histogram/percentile summaries
   - [x] Configurable thresholds
 - [x] Membership inference attack
   - [x] Train classifier to detect synthetic vs real
   - [x] Measure attack success rate
 - [x] Attribute inference attack
   - [x] Predict sensitive attributes using synthetic-only training
   - [x] Report accuracy/F1 or RMSE/R²
 - [x] Attribute inference attack
   - [x] Predict sensitive attributes using synthetic-only training
   - [x] Report accuracy/F1 or RMSE/R²
 - [x] Differential privacy tools
   - [x] Gaussian, Laplace, Exponential mechanisms
   - [x] Parameter conversion (CDP <-> ADP)

**Priority:** Medium
**Estimated Effort:** 3-4 weeks

---

### Phase 11: ML Utility Metrics ✅ (COMPLETED)

**Status:** ✅ Implemented in v1.1.0

- [x] Train-on-Synthetic, Test-on-Real (TSTR)
  - [x] RandomForest baseline
  - [x] Compare to Train-on-Real baseline
- [ ] Train-on-Real, Test-on-Synthetic (TRTS)
  - [ ] Inverse evaluation (future enhancement)
- [ ] Support multiple models
  - [ ] XGBoost (future enhancement)
  - [ ] Logistic Regression (future enhancement)
  - [ ] Neural Networks (future enhancement)
- [ ] Cross-validation
  - [ ] Confidence intervals (future enhancement)
  - [ ] Statistical significance tests (future enhancement)

**Current Implementation:**
- `sdeval/metrics/ml_efficacy.py`
- Supports classification (F1, Accuracy) and regression (MAE, MSE, R2)
- Requires `target_column` in config
- Auto-detects task type
- Returns prefixed metrics: `ml_efficacy_*`

---

### Phase 12: Constraint Checking ✅ (COMPLETED)

**Status:** ✅ Implemented in v1.1.0

- [x] Equality constraints
  - [x] Column must equal specific value
  - [x] Simple categorical constraints (e.g., `education=Bachelors`)
  - [x] Multiple constraints with AND logic
- [x] Expression constraints
  - [x] Pandas eval expressions
- [x] Share constraints
  - [x] Population share targets
- [x] Mean/Min/Max constraints
  - [x] Statistical bounds
- [ ] Conditional constraints
  - [ ] IF-THEN rules (future enhancement)

**Current Implementation:**
- `sdeval/metrics/constraints.py`
- `compute_constraint_satisfaction_rate()` - Single dataset evaluation
- `compute_constraint_support()` - Real vs synthetic comparison
- Supports categorical constraints with simple syntax
- Complex constraints via pandas eval expressions
- Returns `constraints_*` prefixed metrics
- Example: `example_constraint_evaluation.py`

---

### Phase 13: Performance Optimization (Future)

- [ ] Parallel metric execution
  - [ ] Use concurrent.futures
  - [ ] Metrics are independent
- [ ] Smart sampling
  - [ ] Full data for cheap metrics
  - [ ] Sampled data for expensive metrics
- [ ] Streaming CSV reader
  - [ ] For very large datasets
  - [ ] Chunked processing
- [ ] Caching
  - [ ] Cache intermediate results
  - [ ] Avoid recomputation

**Priority:** Low (unless performance issues)
**Estimated Effort:** 1-2 weeks

---

### Phase 14: Advanced Visualization (Remaining)

- [ ] Interactive/HTML dashboards
  - [ ] Plotly/Streamlit drill-downs
  - [ ] One-click HTML report bundling all metrics/plots

**Priority:** Medium  
**Estimated Effort:** 2-3 weeks

### Phase 16: Additional Features (Future)

- [ ] Plausibility scoring
  - [ ] Autoregressive model
  - [ ] Requires separate training
  - [ ] Complex implementation (see old code)
- [ ] Differential privacy verification
  - [ ] Placeholder only
  - [ ] Requires generator metadata
- [ ] Fairness metrics
  - [ ] Demographic parity
  - [ ] Equalized odds
  - [ ] Requires sensitive attribute specification
- [ ] Causality preservation
  - [ ] Structural causal models
  - [ ] Research-level feature

**Priority:** Low (research features)
**Estimated Effort:** Variable

---

## 📊 Current Metrics Reference

### Statistical Fidelity (5 metrics)
1. **statistical_alpha_precision** - Synthetic categories in real data (0-1, higher better)
2. **statistical_beta_recall** - Real categories in synthetic data (0-1, higher better)
3. **statistical_mean_abs_mean_diff** - Column mean differences (lower better)
4. **statistical_mean_abs_std_diff** - Column std differences (lower better)
5. **statistical_avg_wasserstein** - Distribution distance (lower better)

### Coverage & Diversity (4 metrics)
1. **coverage_uniqueness_ratio** - Unique rows fraction (0-1, higher better)
2. **coverage_rare_category_retention** - Rare categories preserved (0-1, higher better)
3. **coverage_missing_category_ratio** - Missing categories (0-1, lower better)
4. **coverage_missingness_delta** - Null rate difference (lower better)

### Privacy (3 metrics)
1. **privacy_dcr** - Distance to closest record (0-1, lower better)
2. **privacy_nndr** - Nearest neighbor ratio (higher better)
3. **privacy_knn_distance** - Average nearest distance (higher better)

### ML Efficacy (variable metrics)
- **ml_efficacy_enabled** - Whether ML evaluation ran
- **ml_efficacy_task_type** - Classification or regression
- **ml_efficacy_accuracy** - Classification accuracy (classification only)
- **ml_efficacy_f1_macro** - Macro F1 score (classification only)
- **ml_efficacy_rmse** - Root mean squared error (regression only)
- **ml_efficacy_mae** - Mean absolute error (regression only)
- **ml_efficacy_r2** - R² score (regression only)

### Constraint Satisfaction (variable metrics)
- **constraints_enabled** - Whether constraints were evaluated
- **constraints_total_rules** - Total number of rules
- **constraints_passed_rules** - Number of rules that passed
- **constraints_hard_failures** - Number of hard constraint failures
- **constraints_rule_details** - Detailed results per rule

**Note:** All metric keys now use standardized `{category}_{metric_name}` naming convention as of v1.1.0

---

## 🎯 Recommended Next Steps

1. **Advanced Statistical & Coverage Metrics (Phases 8 & 9)**  
   - Add KS/Chi-square per-column reporting, mutual information preservation, pairwise/triwise contingency tests, and frequency-rank correlation.
2. **Advanced Privacy Metrics (Phase 10)**  
   - Implement k-anonymity, membership/attribute-inference attacks, and richer DCR analyses.
3. **Performance Improvements (Phase 13)**  
   - Parallel metric execution, smart sampling, streaming CSV reads, caching.
4. **Interactive Visualizations (Phase 14 remaining)**  
   - Generate Plotly/HTML dashboards combining metrics, plots, and drill-down tables.
5. **Research Features (Phase 16)**  
   - Fairness, differential privacy verification hooks, and optional causality metrics for specialized users.

The current version (1.3.0) is production-ready; pursue the items above based on project priorities.

---

## 📝 Contributing

To add features from this roadmap:

1. Pick a phase from the list above
2. Create a feature branch: `git checkout -b feature/phase-X`
3. Write tests first (TDD approach)
4. Implement the feature
5. Update documentation
6. Submit pull request

See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidelines.

---

## 📧 Questions?

- **Usage Questions:** See [README.md](./README.md)
- **Feature Requests:** Open a GitHub issue
- **Bug Reports:** Open a GitHub issue with test case
- **General Discussion:** GitHub Discussions

---

**Current Status: Production Ready with Visual Diagnostics & CLI ✅**
**Last Updated:** January 16, 2025
**Version:** 1.3.0
