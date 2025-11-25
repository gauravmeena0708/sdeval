#!/usr/bin/env python
"""
Example script for evaluating synthetic data using the Adult dataset.

This script demonstrates the complete workflow:
1. Load real and synthetic data
2. Detect column types
3. Compute all metrics
4. Generate visualizations
5. Save results to JSON

Usage:
    python example_evaluation.py
"""

import json
from pathlib import Path

from sdeval.data_loader import load_csv, detect_column_types
from sdeval.metrics.statistical import compute_statistical_metrics
from sdeval.metrics.coverage import compute_coverage_metrics
from sdeval.metrics.privacy import compute_privacy_metrics
from sdeval.visualization import create_distribution_plots


def main():
    """Run complete evaluation pipeline."""

    print("=" * 60)
    print("Synthetic Data Evaluation - Adult Dataset Example")
    print("=" * 60)
    print()

    # Configuration
    real_data_path = Path("datasets/adult/train.csv")
    synthetic_data_path = Path("datasets/adult/test.csv")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Load data
    print("📂 Loading datasets...")
    real_df = load_csv(real_data_path)
    synthetic_df = load_csv(synthetic_data_path)
    print(f"   Real data: {len(real_df):,} rows, {len(real_df.columns)} columns")
    print(f"   Synthetic data: {len(synthetic_df):,} rows, {len(synthetic_df.columns)} columns")
    print()

    # Step 2: Auto-detect column types
    print("🔍 Detecting column types...")
    col_types = detect_column_types(real_df)
    print(f"   Numerical columns ({len(col_types['numerical_columns'])}): {col_types['numerical_columns'][:3]}...")
    print(f"   Categorical columns ({len(col_types['categorical_columns'])}): {col_types['categorical_columns'][:3]}...")
    print()

    # Step 3: Compute statistical fidelity metrics
    print("📊 Computing statistical fidelity metrics...")
    stats = compute_statistical_metrics(
        real_df, synthetic_df,
        col_types['numerical_columns'],
        col_types['categorical_columns']
    )
    print(f"   ✓ Alpha Precision: {stats['statistical_alpha_precision']:.3f}")
    print(f"   ✓ Beta Recall: {stats['statistical_beta_recall']:.3f}")
    print(f"   ✓ Mean Absolute Difference: {stats['statistical_mean_abs_mean_diff']:.3f}")
    print(f"   ✓ Std Absolute Difference: {stats['statistical_mean_abs_std_diff']:.3f}")
    print(f"   ✓ Wasserstein Distance: {stats['statistical_avg_wasserstein']:.3f}")
    print()

    # Step 4: Compute coverage metrics
    print("📈 Computing coverage & diversity metrics...")
    coverage = compute_coverage_metrics(
        real_df, synthetic_df,
        col_types['categorical_columns']
    )
    print(f"   ✓ Uniqueness Ratio: {coverage['coverage_uniqueness_ratio']:.3f}")
    print(f"   ✓ Rare Category Retention: {coverage['coverage_rare_category_retention']:.3f}")
    print(f"   ✓ Missing Category Ratio: {coverage['coverage_missing_category_ratio']:.3f}")
    print(f"   ✓ Missingness Delta: {coverage['coverage_missingness_delta']:.3f}")
    print()

    # Step 5: Compute privacy metrics (using samples for speed)
    print("🔒 Computing privacy metrics (using samples)...")
    real_sample = real_df.sample(n=min(1000, len(real_df)), random_state=42)
    syn_sample = synthetic_df.sample(n=min(500, len(synthetic_df)), random_state=42)

    privacy = compute_privacy_metrics(
        real_sample, syn_sample,
        col_types['numerical_columns']
    )
    print(f"   ✓ DCR Rate: {privacy['dcr_rate']:.3f}")
    print(f"   ✓ NNDR Mean: {privacy['nndr_mean']:.3f}")
    print(f"   ✓ Mean k-NN Distance: {privacy['mean_knn_distance']:.3f}")
    print()

    # Step 6: Combine all results
    results = {
        'metadata': {
            'real_data_path': str(real_data_path),
            'synthetic_data_path': str(synthetic_data_path),
            'real_data_rows': len(real_df),
            'synthetic_data_rows': len(synthetic_df),
            'numerical_columns': col_types['numerical_columns'],
            'categorical_columns': col_types['categorical_columns']
        },
        'statistical': stats,
        'coverage': coverage,
        'privacy': privacy
    }

    # Step 7: Save results as JSON
    print("💾 Saving results...")
    output_json = output_dir / "evaluation_results.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✓ JSON results saved to: {output_json}")
    print()

    # Step 8: Generate visualizations
    print("🎨 Generating distribution plots...")
    viz_path = output_dir / "distributions.png"
    create_distribution_plots(
        real_df, synthetic_df,
        col_types['numerical_columns'][:4],  # First 4 numerical columns
        col_types['categorical_columns'][:4],  # First 4 categorical columns
        viz_path
    )
    print(f"   ✓ Visualization saved to: {viz_path}")
    print()

    # Summary
    print("=" * 60)
    print("✅ Evaluation Complete!")
    print("=" * 60)
    print()
    print("📁 Output files:")
    print(f"   - {output_json}")
    print(f"   - {viz_path}")
    print()
    print("💡 Interpretation Guide:")
    print("   Statistical Fidelity:")
    print("     - Alpha/Beta close to 1.0 = Good category coverage")
    print("     - Low mean/std differences = Similar distributions")
    print("     - Low Wasserstein = Similar overall distributions")
    print()
    print("   Coverage & Diversity:")
    print("     - High uniqueness = Low duplication")
    print("     - High rare retention = Preserves minority groups")
    print("     - Low missing categories = Good coverage")
    print()
    print("   Privacy:")
    print("     - Low DCR = Good privacy (less memorization)")
    print("     - High NNDR = Synthetic is distinct from any single record")
    print("     - High k-NN distance = Good privacy")
    print()


if __name__ == "__main__":
    main()
