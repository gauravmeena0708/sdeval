"""
Test privacy metrics using Adult dataset.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sdeval.data_loader import load_csv, detect_column_types
from sdeval.metrics.privacy import (
    compute_dcr,
    compute_nndr,
    compute_mean_knn_distance,
    compute_privacy_metrics
)


# Test data paths
TRAIN_CSV = Path("datasets/adult/train.csv")
TEST_CSV = Path("datasets/adult/test.csv")


@pytest.fixture
def adult_data():
    """Load Adult dataset for testing."""
    real_df = load_csv(TRAIN_CSV)
    synthetic_df = load_csv(TEST_CSV)
    col_types = detect_column_types(real_df)
    return real_df, synthetic_df, col_types


@pytest.fixture
def small_data():
    """Create small test datasets for privacy tests."""
    real_df = pd.DataFrame({
        'num1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'num2': [10.0, 20.0, 30.0, 40.0, 50.0],
        'cat1': ['A', 'B', 'C', 'D', 'E']
    })
    synthetic_df = pd.DataFrame({
        'num1': [1.1, 2.1, 10.0, 11.0],
        'num2': [10.5, 20.5, 100.0, 110.0],
        'cat1': ['A', 'B', 'X', 'Y']
    })
    return real_df, synthetic_df


class TestDCR:
    """Test Distance to Closest Record (DCR)."""

    def test_dcr_identical_data(self, small_data):
        """Test DCR when synthetic data is identical to real data."""
        real_df, _ = small_data
        synthetic_df = real_df.copy()

        dcr = compute_dcr(real_df, synthetic_df, ['num1', 'num2'], threshold=1e-8)
        assert dcr > 0.5  # Most records should be very close

    def test_dcr_very_different_data(self, small_data):
        """Test DCR when synthetic data is very different from real data."""
        real_df = pd.DataFrame({
            'num1': [1.0, 2.0, 3.0],
            'num2': [10.0, 20.0, 30.0]
        })
        synthetic_df = pd.DataFrame({
            'num1': [100.0, 200.0, 300.0],
            'num2': [1000.0, 2000.0, 3000.0]
        })

        dcr = compute_dcr(real_df, synthetic_df, ['num1', 'num2'], threshold=1e-8)
        assert dcr == 0.0  # No records should be close

    def test_dcr_adult_dataset(self, adult_data):
        """Test DCR on Adult dataset."""
        real_df, synthetic_df, col_types = adult_data

        dcr = compute_dcr(real_df, synthetic_df, col_types['numerical_columns'])

        assert 0.0 <= dcr <= 1.0
        assert np.isfinite(dcr)

    def test_dcr_with_threshold(self, small_data):
        """Test DCR with different thresholds."""
        real_df, synthetic_df = small_data

        dcr_strict = compute_dcr(real_df, synthetic_df, ['num1', 'num2'], threshold=1e-10)
        dcr_loose = compute_dcr(real_df, synthetic_df, ['num1', 'num2'], threshold=10.0)

        assert dcr_strict <= dcr_loose  # Looser threshold should have higher DCR


class TestNNDR:
    """Test Nearest Neighbor Distance Ratio (NNDR)."""

    def test_nndr_basic(self, small_data):
        """Test basic NNDR computation."""
        real_df, synthetic_df = small_data

        nndr = compute_nndr(real_df, synthetic_df, ['num1', 'num2'])

        assert nndr > 0.0
        assert np.isfinite(nndr)

    def test_nndr_identical_data(self, small_data):
        """Test NNDR when synthetic is identical to real."""
        real_df, _ = small_data
        synthetic_df = real_df.copy()

        nndr = compute_nndr(real_df, synthetic_df, ['num1', 'num2'])

        # When identical, nearest neighbor has distance 0, so NNDR will be 0
        assert np.isfinite(nndr)
        assert nndr >= 0.0  # Can be 0 for identical data

    def test_nndr_adult_dataset(self, adult_data):
        """Test NNDR on Adult dataset."""
        real_df, synthetic_df, col_types = adult_data

        # Use a sample for speed
        real_sample = real_df.sample(n=min(1000, len(real_df)), random_state=42)
        syn_sample = synthetic_df.sample(n=min(500, len(synthetic_df)), random_state=42)

        nndr = compute_nndr(real_sample, syn_sample, col_types['numerical_columns'])

        assert nndr > 0.0
        assert np.isfinite(nndr)


class TestMeanKNNDistance:
    """Test mean k-NN distance."""

    def test_mean_knn_distance_basic(self, small_data):
        """Test basic mean k-NN distance computation."""
        real_df, synthetic_df = small_data

        dist = compute_mean_knn_distance(real_df, synthetic_df, ['num1', 'num2'])

        assert dist >= 0.0
        assert np.isfinite(dist)

    def test_mean_knn_distance_identical(self, small_data):
        """Test mean k-NN when synthetic is identical to real."""
        real_df, _ = small_data
        synthetic_df = real_df.copy()

        dist = compute_mean_knn_distance(real_df, synthetic_df, ['num1', 'num2'])

        assert dist >= 0.0
        assert np.isfinite(dist)

    def test_mean_knn_distance_far_apart(self):
        """Test mean k-NN when data is far apart."""
        real_df = pd.DataFrame({
            'num1': [1.0, 2.0, 3.0],
            'num2': [1.0, 2.0, 3.0]
        })
        synthetic_df = pd.DataFrame({
            'num1': [100.0, 200.0, 300.0],
            'num2': [100.0, 200.0, 300.0]
        })

        dist = compute_mean_knn_distance(real_df, synthetic_df, ['num1', 'num2'])

        assert dist > 10.0  # Should be large distance

    def test_mean_knn_distance_adult_dataset(self, adult_data):
        """Test mean k-NN on Adult dataset."""
        real_df, synthetic_df, col_types = adult_data

        # Use a sample for speed
        real_sample = real_df.sample(n=min(1000, len(real_df)), random_state=42)
        syn_sample = synthetic_df.sample(n=min(500, len(synthetic_df)), random_state=42)

        dist = compute_mean_knn_distance(real_sample, syn_sample, col_types['numerical_columns'])

        assert dist >= 0.0
        assert np.isfinite(dist)


class TestPrivacyMetricsIntegration:
    """Test the main compute_privacy_metrics function."""

    def test_compute_all_metrics(self, adult_data):
        """Test computing all privacy metrics at once."""
        real_df, synthetic_df, col_types = adult_data

        # Use samples for speed
        real_sample = real_df.sample(n=min(1000, len(real_df)), random_state=42)
        syn_sample = synthetic_df.sample(n=min(500, len(synthetic_df)), random_state=42)

        metrics = compute_privacy_metrics(
            real_sample,
            syn_sample,
            col_types['numerical_columns']
        )

        # Check all expected metrics are present
        assert 'dcr_rate' in metrics
        assert 'nndr_mean' in metrics
        assert 'mean_knn_distance' in metrics

        # Check all values are valid
        for key, value in metrics.items():
            assert np.isfinite(value)
            assert value >= 0.0

        # DCR should be between 0 and 1
        assert 0.0 <= metrics['dcr_rate'] <= 1.0

    def test_compute_with_empty_numerical_columns(self, small_data):
        """Test privacy metrics with no numerical columns."""
        real_df, synthetic_df = small_data

        metrics = compute_privacy_metrics(real_df, synthetic_df, [])

        # Should return default values or handle gracefully
        assert isinstance(metrics, dict)
        assert len(metrics) == 3
