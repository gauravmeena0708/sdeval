"""
Test statistical fidelity metrics using Adult dataset.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sdeval.data_loader import load_csv, detect_column_types
from sdeval.metrics.statistical import (
    compute_alpha_precision,
    compute_beta_recall,
    compute_mean_absolute_difference,
    compute_std_absolute_difference,
    compute_wasserstein_distance,
    compute_statistical_metrics
)


# Test data paths
TRAIN_CSV = Path("datasets/adult/train.csv")
TEST_CSV = Path("datasets/adult/test.csv")


@pytest.fixture
def adult_data():
    """Load Adult dataset for testing."""
    real_df = load_csv(TRAIN_CSV)
    synthetic_df = load_csv(TEST_CSV)  # Using test as "synthetic" for testing
    col_types = detect_column_types(real_df)
    return real_df, synthetic_df, col_types


class TestAlphaPrecision:
    """Test alpha precision (synthetic values present in real data)."""

    def test_alpha_precision_perfect_match(self):
        """Test alpha when synthetic is subset of real."""
        real_df = pd.DataFrame({'cat': ['A', 'B', 'C', 'D']})
        synthetic_df = pd.DataFrame({'cat': ['A', 'B', 'C']})

        alpha = compute_alpha_precision(real_df, synthetic_df, ['cat'])
        assert alpha == 1.0  # All synthetic values exist in real

    def test_alpha_precision_no_match(self):
        """Test alpha when synthetic has no real values."""
        real_df = pd.DataFrame({'cat': ['A', 'B', 'C']})
        synthetic_df = pd.DataFrame({'cat': ['X', 'Y', 'Z']})

        alpha = compute_alpha_precision(real_df, synthetic_df, ['cat'])
        assert alpha == 0.0  # No synthetic values exist in real

    def test_alpha_precision_partial_match(self):
        """Test alpha with partial overlap."""
        real_df = pd.DataFrame({'cat': ['A', 'B', 'C']})
        synthetic_df = pd.DataFrame({'cat': ['A', 'B', 'X', 'Y']})

        alpha = compute_alpha_precision(real_df, synthetic_df, ['cat'])
        assert 0.0 < alpha < 1.0  # Partial match

    def test_alpha_precision_adult_dataset(self, adult_data):
        """Test alpha on Adult dataset."""
        real_df, synthetic_df, col_types = adult_data
        alpha = compute_alpha_precision(real_df, synthetic_df, col_types['categorical_columns'])

        assert 0.0 <= alpha <= 1.0
        assert alpha > 0.5  # Adult train/test should have high overlap


class TestBetaRecall:
    """Test beta recall (real values covered by synthetic data)."""

    def test_beta_recall_perfect_coverage(self):
        """Test beta when synthetic covers all real values."""
        real_df = pd.DataFrame({'cat': ['A', 'B', 'C']})
        synthetic_df = pd.DataFrame({'cat': ['A', 'B', 'C', 'D']})

        beta = compute_beta_recall(real_df, synthetic_df, ['cat'])
        assert beta == 1.0  # All real values are covered

    def test_beta_recall_no_coverage(self):
        """Test beta when synthetic covers no real values."""
        real_df = pd.DataFrame({'cat': ['A', 'B', 'C']})
        synthetic_df = pd.DataFrame({'cat': ['X', 'Y', 'Z']})

        beta = compute_beta_recall(real_df, synthetic_df, ['cat'])
        assert beta == 0.0  # No coverage

    def test_beta_recall_adult_dataset(self, adult_data):
        """Test beta on Adult dataset."""
        real_df, synthetic_df, col_types = adult_data
        beta = compute_beta_recall(real_df, synthetic_df, col_types['categorical_columns'])

        assert 0.0 <= beta <= 1.0
        assert beta > 0.5  # Adult train/test should have good coverage


class TestMeanAbsoluteDifference:
    """Test mean absolute difference for numerical columns."""

    def test_mean_diff_identical_data(self):
        """Test mean diff when data is identical."""
        real_df = pd.DataFrame({'num': [1, 2, 3, 4, 5]})
        synthetic_df = pd.DataFrame({'num': [1, 2, 3, 4, 5]})

        diff = compute_mean_absolute_difference(real_df, synthetic_df, ['num'])
        assert diff == 0.0  # Identical means

    def test_mean_diff_different_data(self):
        """Test mean diff with different data."""
        real_df = pd.DataFrame({'num': [1, 2, 3, 4, 5]})  # mean = 3
        synthetic_df = pd.DataFrame({'num': [6, 7, 8, 9, 10]})  # mean = 8

        diff = compute_mean_absolute_difference(real_df, synthetic_df, ['num'])
        assert diff == 5.0  # |3 - 8| = 5

    def test_mean_diff_adult_dataset(self, adult_data):
        """Test mean diff on Adult dataset."""
        real_df, synthetic_df, col_types = adult_data
        diff = compute_mean_absolute_difference(real_df, synthetic_df, col_types['numerical_columns'])

        assert diff >= 0.0
        assert np.isfinite(diff)


class TestStdAbsoluteDifference:
    """Test standard deviation absolute difference."""

    def test_std_diff_identical_data(self):
        """Test std diff when data is identical."""
        real_df = pd.DataFrame({'num': [1, 2, 3, 4, 5]})
        synthetic_df = pd.DataFrame({'num': [1, 2, 3, 4, 5]})

        diff = compute_std_absolute_difference(real_df, synthetic_df, ['num'])
        assert diff == 0.0

    def test_std_diff_adult_dataset(self, adult_data):
        """Test std diff on Adult dataset."""
        real_df, synthetic_df, col_types = adult_data
        diff = compute_std_absolute_difference(real_df, synthetic_df, col_types['numerical_columns'])

        assert diff >= 0.0
        assert np.isfinite(diff)


class TestWassersteinDistance:
    """Test Wasserstein distance for distributions."""

    def test_wasserstein_identical_distributions(self):
        """Test Wasserstein when distributions are identical."""
        real_df = pd.DataFrame({'num': [1, 2, 3, 4, 5]})
        synthetic_df = pd.DataFrame({'num': [1, 2, 3, 4, 5]})

        dist = compute_wasserstein_distance(real_df, synthetic_df, ['num'])
        assert dist == 0.0  # Identical distributions

    def test_wasserstein_different_distributions(self):
        """Test Wasserstein with different distributions."""
        real_df = pd.DataFrame({'num': [1, 2, 3]})
        synthetic_df = pd.DataFrame({'num': [4, 5, 6]})

        dist = compute_wasserstein_distance(real_df, synthetic_df, ['num'])
        assert dist > 0.0  # Different distributions

    def test_wasserstein_adult_dataset(self, adult_data):
        """Test Wasserstein on Adult dataset."""
        real_df, synthetic_df, col_types = adult_data
        dist = compute_wasserstein_distance(real_df, synthetic_df, col_types['numerical_columns'])

        assert dist >= 0.0
        assert np.isfinite(dist)


class TestStatisticalMetricsIntegration:
    """Test the main compute_statistical_metrics function."""

    def test_compute_all_metrics(self, adult_data):
        """Test computing all statistical metrics at once."""
        real_df, synthetic_df, col_types = adult_data

        metrics = compute_statistical_metrics(
            real_df,
            synthetic_df,
            col_types['numerical_columns'],
            col_types['categorical_columns']
        )

        # Check all expected metrics are present
        assert 'statistical_alpha_precision' in metrics
        assert 'statistical_beta_recall' in metrics
        assert 'statistical_mean_abs_mean_diff' in metrics
        assert 'statistical_mean_abs_std_diff' in metrics
        assert 'statistical_avg_wasserstein' in metrics

        # Check all values are valid
        for key, value in metrics.items():
            assert np.isfinite(value)
            assert value >= 0.0

        # Check alpha and beta are between 0 and 1
        assert 0.0 <= metrics['statistical_alpha_precision'] <= 1.0
        assert 0.0 <= metrics['statistical_beta_recall'] <= 1.0
