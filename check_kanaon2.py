import pandas as pd
from pycanon import anonymity

# --- Part 1: Your Original pycanon Analysis ---
# (This part is largely unchanged and correct)

# 1. Load the Adult dataset
url = "datasets/adult/train.csv"
try:
    data = pd.read_csv(
        url,
        sep=r',\s*',  # Regex separator for comma + optional space
        engine='python',
        na_values="?"  # Handle '?' as missing values
    )
    
    # Clean column names (a common issue with this dataset)
    data.columns = data.columns.str.strip()
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please check your file path and ensure the file exists.")
    exit()

# Drop rows with missing values
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

# 2. Define Quasi-Identifiers (QI)
# Using hyphens, which is standard for the headers in train.csv
quasi_identifiers = [
    'age',
    'workclass',
    'education',
    'marital.status',
    'occupation',
    'race',
    'sex',
    'native.country'
]

# 3. Define Sensitive Attributes (SA)
sensitive_attribute = ['income']

# 4. Calculate k-anonymity
print("--- Part 1: pycanon Privacy Analysis (Original Data) ---")
try:
    k = anonymity.k_anonymity(data, quasi_identifiers)
    print(f"K-Anonymity: {k}-anonymous")
    
    l = anonymity.l_diversity(data, quasi_identifiers, sensitive_attribute)
    print(f"L-Diversity: {l}-diverse")
    
    # T-Closeness calculation
    data_for_t = data.copy()
    data_for_t['income_numeric'] = data_for_t['income'].astype('category').cat.codes
    numeric_sensitive_attribute = ['income_numeric']
    
    t = anonymity.t_closeness(data_for_t, quasi_identifiers, numeric_sensitive_attribute)
    print(f"T-Closeness: {t}-close")

except KeyError as e:
    print(f"\n[Error] A column name was not found: {e}")
    print("Please check that your 'quasi_identifiers' list matches the column headers in train.csv exactly.")
except Exception as e:
    print(f"\nAn error occurred during pycanon calculation: {e}")

print("\n" + "="*60 + "\n")


# --- Part 2: Synthetic Data Privacy "Parameters" (SDMetrics) ---
# This is the new section to include the "other params" we discussed.

print("--- Part 2: SDMetrics Privacy Analysis (Synthetic Data) ---")
print("This part requires 'sdv' and 'sdmetrics'.")

try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    from sdmetrics.reports.single_table import DiagnosticReport, QualityReport
    from sdmetrics.single_table.privacy import (
        CategoricalCAP,
        CategoricalEnsemble,
        NumericalMLP,
        DCRBaselineProtection,
        DCROverfittingProtection,
        DisclosureProtection
    )
    from sdmetrics.single_table import NewRowSynthesis

    # 1. Generate Synthetic Data
    # We will train a simple model on the real data
    print("Training synthetic data model (CTGAN)...")

    # Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    model = CTGANSynthesizer(metadata)
    model.fit(data)
    
    print("Generating synthetic data sample...")
    synthetic_data = model.sample(num_rows=len(data))

    print("Synthetic data generated. Now calculating privacy metrics...\n")

    # 2. Prepare Metadata dictionary for SDMetrics
    metadata_dict = metadata.to_dict()

    # 3. Calculate New Row Synthesis
    # This checks if synthetic records are novel or just copies of real records.
    print("--- New Row Synthesis ---")
    new_row_score = NewRowSynthesis.compute(
        real_data=data,
        synthetic_data=synthetic_data,
        metadata=metadata_dict
    )
    print(f"New Row Synthesis Score: {new_row_score:.4f}")
    print("(Score closer to 1.0 means more novel synthetic rows, which is better for privacy)\n")

    # 4. Calculate Distance to Closest Record (DCR) - Baseline Protection
    # This checks if synthetic records are too similar to real records.
    print("--- DCR Baseline Protection ---")
    try:
        dcr_baseline = DCRBaselineProtection.compute(
            real_data=data,
            synthetic_data=synthetic_data,
            metadata=metadata_dict
        )
        print(f"DCR Baseline Protection Score: {dcr_baseline:.4f}")
        print("(Higher scores indicate better protection from identifying original records)\n")
    except Exception as e:
        print(f"Could not compute DCR Baseline Protection: {e}\n")

    # 5. Calculate DCR Overfitting Protection
    print("--- DCR Overfitting Protection ---")
    try:
        # DCROverfittingProtection requires additional synthetic test data
        # Generate another sample for the test set
        synthetic_test = model.sample(num_rows=len(data))
        dcr_overfitting = DCROverfittingProtection.compute(
            real_data=data,
            synthetic_data=synthetic_data,
            synthetic_test_data=synthetic_test,
            metadata=metadata_dict
        )
        print(f"DCR Overfitting Protection Score: {dcr_overfitting:.4f}")
        print("(Higher scores indicate the model didn't just memorize the training data)\n")
    except Exception as e:
        print(f"Could not compute DCR Overfitting Protection: {e}\n")

    # 6. Calculate Categorical CAP (Privacy Metric)
    # CAP measures privacy for categorical sensitive attributes
    print("--- Categorical CAP (Privacy for Sensitive Attributes) ---")
    try:
        # Use income as the sensitive attribute
        cap_score = CategoricalCAP.compute(
            real_data=data,
            synthetic_data=synthetic_data,
            key_fields=['age', 'workclass', 'education', 'marital.status', 'occupation', 'race', 'sex', 'native.country'],
            sensitive_fields=['income']
        )
        print(f"Categorical CAP Score: {cap_score:.4f}")
        print("(Score closer to 0.0 means better privacy protection for sensitive attributes)\n")
    except Exception as e:
        print(f"Could not compute Categorical CAP: {e}\n")

    # 7. (Optional) Run a simple diagnostic report
    # This checks for basic data integrity.
    print("--- Diagnostic Report (Basic Integrity) ---")
    try:
        diagnostic = DiagnosticReport()
        diagnostic.generate(real_data=data, synthetic_data=synthetic_data, metadata=metadata_dict)
        # Get the overall score using get_score() method
        overall_score = diagnostic.get_score()
        print(f"Overall Diagnostic Score: {overall_score:.2f}%")

        # Get detailed properties (individual test results)
        properties = diagnostic.get_properties()
        print("\nDiagnostic Details:")
        for prop_name, prop_data in properties.items():
            score = prop_data.get('Score', 'N/A')
            print(f"  {prop_name}: {score}")
    except Exception as e:
        print(f"Could not generate diagnostic report: {e}\n")

except ImportError as ie:
    print(f"Could not import required libraries: {ie}")
    print("Please install them with: pip install sdv sdmetrics")
except Exception as e:
    print(f"An error occurred during synthetic privacy calculation: {e}")
    import traceback
    traceback.print_exc()