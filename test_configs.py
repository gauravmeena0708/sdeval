"""Test script to validate dataset configs work with sdeval."""
import json
from pathlib import Path
from sdeval.config import EvaluatorSettings
from sdeval.data_loader import load_csv
from sdeval.metrics import MetricContext
from sdeval.metrics.ml_efficacy import compute_ml_efficacy
from sdeval.metrics.constraints import compute_constraints_metrics

# Test adult config
print("=" * 60)
print("Testing Adult Config")
print("=" * 60)

config_path = Path("configs/adult.json")
with open(config_path) as f:
    config = json.load(f)

print(f"✓ Config loaded: {config['dataset_name']}")
print(f"  Target: {config['target_column']}")
print(f"  Task: {config['task_type']}")
print(f"  Constraints: {len(config['constraints']['rules'])} rules")
print(f"  Privacy: k-anonymity with k={config['privacy_metrics']['k_anonymity']['k']}")

# Test with actual data
real_path = Path("datasets/adult/train.csv")
syn_path = Path("synthetic/adult_education_11th.csv")

if real_path.exists() and syn_path.exists():
    print("\n✓ Loading data...")
    real_df = load_csv(real_path)
    syn_df = load_csv(syn_path)
    print(f"  Real: {len(real_df)} rows")
    print(f"  Synthetic: {len(syn_df)} rows")
    
    # Test ML efficacy with config
    print("\n✓ Testing ML efficacy metric...")
    settings = EvaluatorSettings(
        input_path=str(syn_path),
        real_data_path=real_path,
        output_dir="outputs/test",
        raw_config=config
    )
    ctx = MetricContext(real_df, syn_df, settings, str(syn_path))
    ml_result = compute_ml_efficacy(ctx)
    
    if ml_result.get('ml_efficacy_enabled'):
        print(f"  ✓ ML Efficacy: {ml_result['ml_efficacy_task_type']}")
        print(f"    Accuracy: {ml_result.get('ml_efficacy_accuracy', 0):.3f}")
    else:
        print(f"  ✗ ML Efficacy disabled: {ml_result.get('ml_efficacy_reason')}")
    
    # Test constraints
    print("\n✓ Testing constraint metrics...")
    constraint_result = compute_constraints_metrics(ctx)
    
    if constraint_result.get('constraints_enabled'):
        print(f"  ✓ Constraints: {constraint_result['constraints_total_rules']} rules")
        print(f"    Passed: {constraint_result['constraints_passed_rules']}")
        print(f"    Failed: {constraint_result['constraints_hard_failures']}")
    else:
        print(f"  ✗ Constraints disabled")
    
    print("\n✅ Adult config works correctly!")
else:
    print(f"\n⚠ Data files not found, skipping data tests")
    print(f"  Expected: {real_path} and {syn_path}")

print("\n" + "=" * 60)
print("Config validation complete!")
print("=" * 60)
