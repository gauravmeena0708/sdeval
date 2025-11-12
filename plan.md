# Plan for Synthetic Data Evaluator

## 1. Objective

To create a comprehensive, command-line based evaluation tool for assessing the quality of synthetic tabular data. This tool will calculate a variety of metrics across different categories, providing a holistic view of the synthetic data's fidelity, utility, and privacy.

## 2. CLI Interface

The tool will be invoked via the command line with the following arguments:

```bash
python -m evaluator.main \
    --input-path <path_to_synthetic_data_csv_or_folder> \
    --training-data-csv-path <path_to_real_data_csv> \
    --output-dir <path_to_output_directory> \
    [--model-path <path_to_pretrained_model_folder>] \
    [--configs <path_to_config_json_file>] \
    [--overwrite]
```

-   `--input-path`: Path to the synthetic data. Can be a single CSV file or a folder containing multiple CSV files.
-   `--training-data-csv-path`: Path to the real training data CSV file. This is essential for comparison.
-   `--output-dir`: Path to the directory where the evaluation results will be saved.
-   `--model-path` (optional): Path to a directory containing a pre-trained model, required for metrics like Plausibility Score.
-   `--configs` (optional): Path to a JSON file for custom configurations (e.g., specifying column types, or parameters for certain metrics).
-   `--overwrite` (optional): If specified, any existing results in the output directory will be overwritten.

## 3. Modular Architecture

The evaluator will be structured in a modular way to allow for easy extension and maintenance.

```
Evaluator/
├── main.py             # Main CLI entry point
├── evaluator.py        # Core Evaluator class
├── metrics/
│   ├── __init__.py
│   ├── statistical.py    # Statistical similarity metrics
│   ├── ml_efficacy.py    # Machine Learning efficacy metrics
│   ├── privacy.py        # Privacy-related metrics
│   └── plausibility.py   # Plausibility score (submodule call)
└── utils.py            # Utility functions
```

-   **`main.py`**: Parses command-line arguments and orchestrates the evaluation process.
-   **`evaluator.py`**: Contains the main `Evaluator` class that loads data, calls the different metric calculators, and saves the results.
-   **`metrics/`**: A package containing different modules for each category of metrics.

## 4. Metric Implementation Details

### 4.1. Statistical Similarity & Diversity

-   **Implementation:** This will be implemented in `metrics/statistical.py`.
-   **Libraries:** `pandas`, `numpy`, `scipy`.
-   **Metrics:**
    -   **Wasserstein Distance:** `scipy.stats.wasserstein_distance`
    -   **Kolmogorov-Smirnov (KS) Test:** `scipy.stats.ks_2samp`
    -   **Jensen-Shannon Divergence:** `scipy.spatial.distance.jensenshannon`
    -   **Mean Squared Error:** `numpy.mean((p - q)**2)`
    -   **Mean Absolute Error:** `numpy.mean(numpy.abs(p - q))`
    -   **KL Divergence:** `scipy.stats.entropy`
    -   **Correlation Difference:** `numpy.linalg.norm`
-   **Submodule Required:** No. These can be implemented directly using standard libraries.

### 4.2. Machine Learning Efficacy

-   **Implementation:** This will be implemented in `metrics/ml_efficacy.py`.
-   **Libraries:** `scikit-learn`, `pandas`, `numpy`.
-   **Metrics:**
    -   **Classification:** `sklearn.metrics.accuracy_score`, `sklearn.metrics.f1_score`
    -   **Regression:** `sklearn.metrics.mean_squared_error`, `sklearn.metrics.r2_score`
-   **Submodule Required:** No. This can be implemented using `scikit-learn`. The module will train a model on the synthetic data and evaluate it on the real data.

### 4.3. Privacy

-   **Implementation:** This will be implemented in `metrics/privacy.py`.
-   **Libraries:** `pandas`, `numpy`, `sklearn`.
-   **Metrics:**
    -   **Distance to Closest Record (DCR):** Can be implemented using `sklearn.neighbors.NearestNeighbors` for efficiency.
-   **Submodule Required:** No.

### 4.4. Plausibility Score

-   **Implementation:** This will be a submodule call from `metrics/plausibility.py`.
-   **Rationale:** The plausibility score requires a trained autoregressive model. The logic for training and scoring is complex and is best kept in a separate submodule, like the one found in the `plausibility` project.
-   **Action:** The `metrics/plausibility.py` module will act as a wrapper that calls the `plausibility.scorer.score_csv_file` function. It will pass the `model-path` and other necessary arguments.

### 4.5. Differential Privacy

-   **Implementation:** This will require a dedicated submodule.
-   **Rationale:** Calculating differential privacy guarantees is a complex process that depends on the specific generative model and its training process. It's not a simple post-processing step.
-   **Action:** A separate submodule, potentially based on the `differential_privacy` directory from the `cuts` project, would be needed. This submodule would need access to the model and its training parameters to calculate the privacy budget (epsilon and delta). This metric is highly model-specific and might not be universally applicable to all synthetic data. For the initial version of the evaluator, we can consider this out of scope or provide a placeholder.

## 5. Workflow

1.  **Initialization:** `main.py` parses arguments.
2.  **Evaluator Instantiation:** An `Evaluator` object is created in `evaluator.py`.
3.  **Data Loading:** The `Evaluator` loads the real and synthetic data. If the input is a folder, it iterates through all CSV files.
4.  **Configuration Loading:** The `Evaluator` loads the configuration file if provided. This file can specify column types (numerical/categorical), the target column for ML efficacy, and other parameters. If not provided, the tool will attempt to infer column types.
5.  **Metric Calculation:** The `Evaluator` calls the different metric modules in the `metrics/` package.
    -   For metrics requiring a model (like plausibility), it will check if `--model-path` is provided.
6.  **Result Aggregation:** The `Evaluator` aggregates the results from all metric modules into a single dictionary.
7.  **Output Saving:** The results are saved to a JSON or CSV file in the specified `--output-dir`.

## 6. Configuration File (`configs.json`)

An example `configs.json` file:

```json
{
    "target_column": "income",
    "task_type": "classification",
    "categorical_columns": [
        "workclass",
        "education",
        "marital-status"
    ],
    "numerical_columns": [
        "age",
        "fnlwgt",
        "hours-per-week"
    ],
    "privacy_metrics": {
        "enabled": true
    },
    "plausibility_metrics": {
        "enabled": true
    }
}
```

This allows for fine-grained control over the evaluation process without cluttering the command line.
