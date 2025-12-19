"""
Example script demonstrating constraint satisfaction rate evaluation.

This script shows how to use the constraint satisfaction rate metrics
to compare real and synthetic data with categorical constraints.
"""
from pathlib import Path
import pandas as pd
from sdeval.metrics.constraints import (
    compute_constraint_satisfaction_rate,
    compute_constraint_support
)

# Load Adult dataset
train_path = Path("datasets/adult/train.csv")
test_path = Path("datasets/adult/test.csv")

real_df = pd.read_csv(train_path)
synthetic_df = pd.read_csv(test_path)

print("=" * 80)
print("CONSTRAINT SATISFACTION RATE EVALUATION - Adult Dataset")
print("=" * 80)
print(f"\nReal data shape: {real_df.shape}")
print(f"Synthetic data shape: {synthetic_df.shape}")
print()

# Example 1: Single categorical constraint from mappings.csv
print("\n" + "=" * 80)
print("Example 1: Single Constraint - education=11th")
print("=" * 80)
constraint1 = "education=11th"
metrics1 = compute_constraint_support(real_df, synthetic_df, constraint1)
print(f"\nConstraint: {constraint1}")
print(f"Real satisfaction rate: {metrics1['real_satisfaction_rate']:.4f} ({metrics1['real_satisfaction_rate']*100:.2f}%)")
print(f"Synthetic satisfaction rate: {metrics1['synthetic_satisfaction_rate']:.4f} ({metrics1['synthetic_satisfaction_rate']*100:.2f}%)")
print(f"Absolute difference: {metrics1['satisfaction_rate_diff']:.4f}")

# Example 2: workclass constraint
print("\n" + "=" * 80)
print("Example 2: Single Constraint - workclass=State-gov")
print("=" * 80)
constraint2 = "workclass=State-gov"
metrics2 = compute_constraint_support(real_df, synthetic_df, constraint2)
print(f"\nConstraint: {constraint2}")
print(f"Real satisfaction rate: {metrics2['real_satisfaction_rate']:.4f} ({metrics2['real_satisfaction_rate']*100:.2f}%)")
print(f"Synthetic satisfaction rate: {metrics2['synthetic_satisfaction_rate']:.4f} ({metrics2['synthetic_satisfaction_rate']*100:.2f}%)")
print(f"Absolute difference: {metrics2['satisfaction_rate_diff']:.4f}")

# Example 3: Multiple constraints (from files.csv)
print("\n" + "=" * 80)
print("Example 3: Multiple Constraints - workclass=State-gov,education=Bachelors")
print("=" * 80)
constraint3 = "workclass=State-gov,education=Bachelors"
metrics3 = compute_constraint_support(real_df, synthetic_df, constraint3)
print(f"\nConstraint: {constraint3}")
print(f"Real satisfaction rate: {metrics3['real_satisfaction_rate']:.4f} ({metrics3['real_satisfaction_rate']*100:.2f}%)")
print(f"Synthetic satisfaction rate: {metrics3['synthetic_satisfaction_rate']:.4f} ({metrics3['synthetic_satisfaction_rate']*100:.2f}%)")
print(f"Absolute difference: {metrics3['satisfaction_rate_diff']:.4f}")

# Example 4: workclass=Private
print("\n" + "=" * 80)
print("Example 4: Single Constraint - workclass=Private")
print("=" * 80)
constraint4 = "workclass=Private"
metrics4 = compute_constraint_support(real_df, synthetic_df, constraint4)
print(f"\nConstraint: {constraint4}")
print(f"Real satisfaction rate: {metrics4['real_satisfaction_rate']:.4f} ({metrics4['real_satisfaction_rate']*100:.2f}%)")
print(f"Synthetic satisfaction rate: {metrics4['synthetic_satisfaction_rate']:.4f} ({metrics4['synthetic_satisfaction_rate']*100:.2f}%)")
print(f"Absolute difference: {metrics4['satisfaction_rate_diff']:.4f}")

# Example 5: Compare satisfaction rates across different workclass values
print("\n" + "=" * 80)
print("Example 5: Comparison Across Different Workclass Values")
print("=" * 80)
workclass_values = ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov']
print(f"\n{'Workclass':<20} {'Real Rate':<12} {'Synth Rate':<12} {'Difference':<12}")
print("-" * 60)
for wc in workclass_values:
    constraint = f"workclass={wc}"
    try:
        metrics = compute_constraint_support(real_df, synthetic_df, constraint)
        print(f"{wc:<20} {metrics['real_satisfaction_rate']:>10.4f}  {metrics['synthetic_satisfaction_rate']:>10.4f}  {metrics['satisfaction_rate_diff']:>10.4f}")
    except Exception as e:
        print(f"{wc:<20} Error: {e}")

# Example 6: Individual dataset constraint satisfaction (not comparison)
print("\n" + "=" * 80)
print("Example 6: Individual Dataset Satisfaction Rates")
print("=" * 80)
constraint6 = "education=Bachelors"
real_rate = compute_constraint_satisfaction_rate(real_df, constraint6)
synth_rate = compute_constraint_satisfaction_rate(synthetic_df, constraint6)
print(f"\nConstraint: {constraint6}")
print(f"Real data satisfaction: {real_rate:.4f} ({real_rate*100:.2f}%)")
print(f"Synthetic data satisfaction: {synth_rate:.4f} ({synth_rate*100:.2f}%)")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)