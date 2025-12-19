import json
from pathlib import Path
from sdeval.data_loader import load_csv, detect_column_types
from sdeval.metrics.statistical import compute_statistical_metrics
from sdeval.metrics.coverage import compute_coverage_metrics
from sdeval.metrics.privacy import compute_privacy_metrics
from sdeval.metrics.constraints import compute_constraint_support
from sdeval.metrics.ml_efficacy import compute_ml_efficacy
from sdeval.metrics import MetricContext
from sdeval.config import EvaluatorSettings

# Paths
real_data_path = Path("datasets/adult/train.csv")
syn_data_path = Path("synthetic/adult_education_11th.csv")
output_dir = Path("outputs/adult_11th")
output_dir.mkdir(parents=True, exist_ok=True)

# Load Data
print(f"Loading real data: {real_data_path}")
real_df = load_csv(real_data_path)

print(f"Loading synthetic data: {syn_data_path}")
syn_df = load_csv(syn_data_path)

# Auto-detect types
col_types = detect_column_types(real_df)
num_cols = col_types['numerical_columns']
cat_cols = col_types['categorical_columns']

print(f"Numerical columns: {len(num_cols)}")
print(f"Categorical columns: {len(cat_cols)}")

# 1. Statistical (Quality)
print("\n--- 1. Statistical Quality ---")
stats = compute_statistical_metrics(
    real_df, syn_df, num_cols, cat_cols
)
print(f"Alpha Precision: {stats['statistical_alpha_precision']:.3f} (CI: {stats.get('statistical_alpha_precision_ci_low', 0):.3f}-{stats.get('statistical_alpha_precision_ci_high', 0):.3f})")
print(f"Beta Recall: {stats['statistical_beta_recall']:.3f}")
print(f"Wasserstein Dist: {stats['statistical_avg_wasserstein']:.3f}")

# 2. Coverage (Diversity)
print("\n--- 2. Coverage (Diversity) ---")
coverage = compute_coverage_metrics(real_df, syn_df, cat_cols)
print(f"Uniqueness Ratio: {coverage['coverage_uniqueness_ratio']:.3f}")
print(f"Rare Category Retention: {coverage['coverage_rare_category_retention']:.3f}")

# 3. Privacy
print("\n--- 3. Privacy ---")
# Use sample for speed if large
privacy = compute_privacy_metrics(
    real_df.sample(n=min(2000, len(real_df)), random_state=42), 
    syn_df.sample(n=min(2000, len(syn_df)), random_state=42), 
    num_cols
)
print(f"DCR at 1e-6: {privacy['privacy_dcr_at_1e-06']:.3f}")
print(f"NNDR: {privacy['privacy_nndr']:.3f}")
print(f"Distance P50: {privacy['privacy_distance_p50']:.3f}")
print(f"Distance P95: {privacy['privacy_distance_p95']:.3f}")

# 4. Constraints
print("\n--- 4. Constraints (education=11th) ---")
constraint_str = "education=11th"
constraint_res = compute_constraint_support(real_df, syn_df, constraint_str)
print(f"Real Support: {constraint_res['real_satisfaction_rate']:.1%}")
print(f"Synthetic Support: {constraint_res['synthetic_satisfaction_rate']:.1%}")
print(f"Difference: {constraint_res['satisfaction_rate_diff']:.3f}")

# 5. ML Efficacy (Downstream Task)
print("\n--- 5. ML Efficacy (Income Classification) ---")
# Mock settings to pass target column
settings = EvaluatorSettings(
    input_path=str(syn_data_path),
    real_data_path=real_data_path,
    output_dir=str(output_dir),
    raw_config={"target_column": "income", "task_type": "classification"}
)
ctx = MetricContext(real_df, syn_df, settings, str(syn_data_path))
ml_res = compute_ml_efficacy(ctx)

if ml_res.get('ml_efficacy_enabled'):
    print(f"Task: {ml_res['ml_efficacy_task_type']}")
    print(f"Accuracy: {ml_res['ml_efficacy_accuracy']:.3f}")
    print(f"F1 Macro: {ml_res['ml_efficacy_f1_macro']:.3f}")
else:
    print(f"ML Efficacy skipped: {ml_res.get('ml_efficacy_reason')}")

# Save full results
full_results = {
    "statistical": stats,
    "coverage": coverage,
    "privacy": privacy,
    "constraints": constraint_res,
    "ml_efficacy": ml_res
}

from sdeval.reporting import write_summary
summary_path = write_summary(str(output_dir), "evaluation", full_results)
print(f"\nFull results saved to {summary_path}")
