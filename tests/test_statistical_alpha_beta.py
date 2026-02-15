import pytest
import pandas as pd
from pathlib import Path

from sdeval.data_loader import load_csv, detect_column_types
from sdeval.metrics.statistical import compute_alpha_precision, compute_beta_recall # Changed import
from sdeval.config import EvaluatorSettings
from sdeval.metrics import MetricContext

# Test data paths
TRAIN_CSV = Path("datasets/adult/train.csv")
TEST_CSV = Path("datasets/adult/test.csv")

@pytest.fixture
def adult_context():
    """Load Adult dataset and create a MetricContext."""
    real_df = load_csv(TRAIN_CSV)
    synthetic_df = load_csv(TEST_CSV) # Using test as "synthetic" for testing
    
    # Provide dummy paths for EvaluatorSettings
    settings = EvaluatorSettings(
        input_path="dummy_input.csv",
        real_data_path=str(TRAIN_CSV), # Use the actual real data path
        output_dir="dummy_output_dir"
    )
    # Detect column types to pass to the metrics
    col_types = detect_column_types(real_df)
    return MetricContext(
        real_df=real_df,
        synthetic_df=synthetic_df,
        settings=settings,
        synthetic_path="mock_path",
        # We need the categorical columns for the test functions, but the MetricContext
        # itself isn't directly passed to `compute_alpha_precision` and `compute_beta_recall`.
        # We'll extract them in the test function using detect_column_types.
    )

def _create_metric_context(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> MetricContext:
    """Helper to create MetricContext for handcrafted data.
    NOTE: This will not be used by the direct calls to compute_alpha_precision and compute_beta_recall,
    but it's left here for consistency if other metrics need it.
    """
    settings = EvaluatorSettings(
        input_path="dummy_input.csv",
        real_data_path="dummy_real.csv",
        output_dir="dummy_output_dir"
    )
    return MetricContext(
        real_df=real_data,
        synthetic_df=synthetic_data,
        settings=settings,
        synthetic_path="mock_path"
    )

class TestAlphaPrecision:
    """Test alpha precision metric from sdeval.metrics.statistical."""
    
    def test_alpha_precision_perfect_match(self):
        """Test alpha when synthetic is a perfect subset of real categories."""
        real_df = pd.DataFrame({'cat': ['A', 'B', 'C', 'D']})
        synthetic_df = pd.DataFrame({'cat': ['A', 'B', 'C']})
        score = compute_alpha_precision(real_df, synthetic_df, ['cat']) # Direct call
        assert score == 1.0

    def test_alpha_precision_no_match(self):
        """Test alpha when synthetic has no categories present in real."""
        real_df = pd.DataFrame({'cat': ['A', 'B', 'C']})
        synthetic_df = pd.DataFrame({'cat': ['X', 'Y', 'Z']})
        score = compute_alpha_precision(real_df, synthetic_df, ['cat']) # Direct call
        assert score == 0.0

    def test_alpha_precision_partial_match(self):
        """Test alpha with partial overlap of categories."""
        real_df = pd.DataFrame({'cat': ['A', 'B', 'C', 'D']})
        synthetic_df = pd.DataFrame({'cat': ['A', 'B', 'X', 'Y']})
        score = compute_alpha_precision(real_df, synthetic_df, ['cat']) # Direct call
        # The expected value for partial match is 2/4 (for 'A', 'B' in synthetic that are in real) = 0.5
        assert score == 0.5


    def test_alpha_precision_on_adult(self, adult_context):
        """Test alpha precision on the adult dataset."""
        # Need to get categorical columns for adult dataset
        col_types = detect_column_types(adult_context.real_df)
        categorical_columns = col_types['categorical_columns']
        score = compute_alpha_precision(adult_context.real_df, adult_context.synthetic_df, categorical_columns) # Direct call
        assert 0.0 <= score <= 1.0
        assert score > 0.5 


class TestBetaRecall:
    """Test beta recall metric from sdeval.metrics.statistical."""

    def test_beta_recall_perfect_coverage(self):
        """Test beta when synthetic covers all real categories."""
        real_df = pd.DataFrame({'cat': ['A', 'B', 'C']})
        synthetic_df = pd.DataFrame({'cat': ['A', 'B', 'C', 'D']})
        score = compute_beta_recall(real_df, synthetic_df, ['cat']) # Direct call
        assert score == 1.0

    def test_beta_recall_no_coverage(self):
        """Test beta when synthetic covers no real categories."""
        real_df = pd.DataFrame({'cat': ['A', 'B', 'C']})
        synthetic_df = pd.DataFrame({'cat': ['X', 'Y', 'Z']})
        score = compute_beta_recall(real_df, synthetic_df, ['cat']) # Direct call
        assert score == 0.0

    def test_beta_recall_partial_coverage(self):
        """Test beta with partial coverage of categories."""
        real_df = pd.DataFrame({'cat': ['A', 'B', 'C', 'D']})
        synthetic_df = pd.DataFrame({'cat': ['A', 'B', 'X', 'Y']})
        score = compute_beta_recall(real_df, synthetic_df, ['cat']) # Direct call
        # The expected value for partial coverage is 2/4 (for 'A', 'B' from real that are in synthetic) = 0.5
        assert score == 0.5

    def test_beta_recall_on_adult(self, adult_context):
        """Test beta recall on the adult dataset."""
        # Need to get categorical columns for adult dataset
        col_types = detect_column_types(adult_context.real_df)
        categorical_columns = col_types['categorical_columns']
        score = compute_beta_recall(adult_context.real_df, adult_context.synthetic_df, categorical_columns) # Direct call
        assert 0.0 <= score <= 1.0
        assert score > 0.5