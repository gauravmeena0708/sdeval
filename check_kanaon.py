import pandas as pd
from pycanon import anonymity

# 1. Load the Adult dataset from the UCI repository
# The dataset does not have headers, so we define them.
url = "datasets/adult/train.csv"
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education.num', 
    'marital.status', 'occupation', 'relationship', 'race', 'sex', 
    'capital.gain', 'capital.loss', 'hours.per.week', 'native.country', 'income'
]



# Read the CSV data
try:
    data = pd.read_csv(
        url,
        sep=r',\s*',  # Regex separator for comma + optional space
        engine='python',
        na_values="?" # Handle '?' as missing values
    )
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please check your internet connection or the dataset URL.")
    exit()

# Drop rows with missing values for this example
data.dropna(inplace=True)

# Reset index to avoid issues with pycanon
data.reset_index(drop=True, inplace=True)

# 2. Define Quasi-Identifiers (QI)
# These are the columns that could be used for re-identification.
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
# This is the sensitive information to be protected.
sensitive_attribute = ['income']

# 4. Calculate k-anonymity
# This function returns the k value for the dataset.
# A k of 1 means at least one record is unique based on the QIs.
try:
    k = anonymity.k_anonymity(data, quasi_identifiers)
    print(f"--- K-Anonymity ---")
    print(f"The dataset is {k}-anonymous for the chosen QIs.")
    print(f"(This means the smallest group of identical QIs has {k} record(s))")
except Exception as e:
    print(f"An error occurred during k-anonymity calculation: {e}")

# 5. (Optional) Calculate other metrics like l-diversity
try:
    l = anonymity.l_diversity(data, quasi_identifiers, sensitive_attribute)
    print(f"\n--- L-Diversity ---")
    print(f"The dataset is {l}-diverse for the sensitive attribute 'income'.")
except Exception as e:
    print(f"\nAn error occurred during l-diversity calculation: {e}")

# 6. (Optional) Calculate t-closeness
try:
    # Note: t-closeness requires the sensitive attribute to be numeric.
    # We must first convert the 'income' column.

    # Create a copy to avoid changing the original data for other calculations
    data_for_t = data.copy()

    # Convert sensitive attribute to numeric codes
    data_for_t['income_numeric'] = data_for_t['income'].astype('category').cat.codes
    numeric_sensitive_attribute = ['income_numeric']

    # We also need to define the type of the sensitive attribute (numeric or categorical)
    t = anonymity.t_closeness(data_for_t, quasi_identifiers, numeric_sensitive_attribute)
    print(f"\n--- T-Closeness ---")
    print(f"The dataset is {t}-close (numerical) for the sensitive attribute 'income'.")
except Exception as e:
    print(f"\nAn error occurred during t-closeness calculation: {e}")