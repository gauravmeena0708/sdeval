# Dataset Configuration Files

This directory contains standard configuration files for common benchmark datasets used with sdeval.

## Available Configs

| Dataset | File | Task Type | Target Column | Description |
|---------|------|-----------|---------------|-------------|
| **Adult** | `adult.json` | Classification | `income` | UCI Adult Income dataset - predict income >50K |
| **Default** | `default.json` | Classification | `default_payment_next_month` | Credit card default prediction |
| **Shopper** | `shopper.json` | Classification | `Revenue` | Online shopping conversion prediction |
| **Beijing** | `beijing.json` | Regression | `pm2.5` | Air quality PM2.5 prediction |
| **Diabetes** | `diabetes.json` | Classification | `Outcome` | Pima Indians diabetes prediction |
| **News** | `news.json` | Regression | `shares` | News article popularity prediction |

## Usage

Use these configs with the `--configs` flag:

```bash
sdeval evaluate \
  --input-path synthetic/adult_synthetic.csv \
  --real-data-csv-path datasets/adult/train.csv \
  --output-dir outputs \
  --configs configs/adult.json \
  --visualize
```

## Config Structure

Each config includes:
- **target_column**: Column to use for ML efficacy evaluation
- **task_type**: `classification` or `regression`
- **privacy_metrics**: Privacy evaluation settings (k-anonymity, DCR thresholds)
- **constraints**: Domain-specific validation rules
- **notes**: Dataset-specific considerations

## Customization

You can modify these configs or create new ones for your datasets. Key sections:

### Privacy Settings
```json
"privacy_metrics": {
  "dcr_thresholds": [1e-6],
  "k_anonymity": {
    "quasi_identifiers": ["age", "sex", "education"],
    "k": 5
  }
}
```

### Constraint Rules
```json
"constraints": {
  "rules": [
    {
      "id": "age_range",
      "type": "expression",
      "expression": "(age >= 18) & (age <= 100)",
      "description": "Age must be realistic"
    }
  ]
}
```

## Notes

- All configs use standardized metric names (26 essential metrics)
- Privacy settings focus on realistic quasi-identifiers
- Constraints enforce domain knowledge and data quality
- Configs are validated at runtime
