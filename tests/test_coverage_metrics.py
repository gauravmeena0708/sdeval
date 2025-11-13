"""
Test coverage and diversity metrics using Adult dataset.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sdeval.data_loader import load_csv, detect_column_types
from sdeval.metrics.coverage import (
    compute_uniqueness_ratio,
    compute_rare_category_retention,
    compute_missing_category_ratio,
    compute_missingness_delta,
    compute_coverage_metrics
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


class TestUniquenessRatio:
    """Test uniqueness ratio metric."""

    def test_uniqueness_all_unique(self):
        """Test with all unique rows."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ratio = compute_uniqueness_ratio(df)
        assert ratio == 1.0

    def test_uniqueness_all_duplicates(self):
        """Test with all duplicate rows."""
        df = pd.DataFrame({'a': [1, 1, 1], 'b': ['x', 'x', 'x']})
        ratio = compute_uniqueness_ratio(df)
        assert ratio == pytest.approx(1.0 / 3.0)

    def test_uniqueness_partial_duplicates(self):
        """Test with some duplicates."""
        df = pd.DataFrame({'a': [1, 1, 2, 3], 'b': ['x', 'x', 'y', 'z']})
        ratio = compute_uniqueness_ratio(df)
        assert 0.0 < ratio < 1.0

    def test_uniqueness_adult_dataset(self, adult_data):
        """Test on Adult dataset."""
        _, synthetic_df, _ = adult_data
        ratio = compute_uniqueness_ratio(synthetic_df)

        assert 0.0 <= ratio <= 1.0
        assert ratio > 0.9  # Adult dataset should have mostly unique rows


class TestRareCategoryRetention:
    """Test rare category retention metric."""

    def test_rare_categories_all_retained(self):
        """Test when all rare categories are retained."""
        # 'A' appears 10 times (10%), 'B' appears 2 times (2% - rare)
        real_df = pd.DataFrame({'cat': ['A'] * 10 + ['B'] * 2})
        synthetic_df = pd.DataFrame({'cat': ['A'] * 5 + ['B'] * 1})

        retention = compute_rare_category_retention(real_df, synthetic_df, ['cat'])
        assert retention == 1.0  # 'B' is retained

    def test_rare_categories_none_retained(self):
        """Test when no rare categories are retained."""
        real_df = pd.DataFrame({'cat': ['A'] * 20 + ['B'] * 1})  # 'B' is rare (< 5%)
        synthetic_df = pd.DataFrame({'cat': ['A'] * 10})  # 'B' missing

        retention = compute_rare_category_retention(real_df, synthetic_df, ['cat'])
        assert retention == 0.0

    def test_rare_categories_no_rare_values(self):
        """Test when there are no rare categories."""
        real_df = pd.DataFrame({'cat': ['A'] * 50 + ['B'] * 50})  # Both common
        synthetic_df = pd.DataFrame({'cat': ['A', 'B']})

        retention = compute_rare_category_retention(real_df, synthetic_df, ['cat'])
        assert retention == 1.0  # No rare categories to retain

    def test_rare_categories_adult_dataset(self, adult_data):
        """Test on Adult dataset."""
        real_df, synthetic_df, col_types = adult_data
        retention = compute_rare_category_retention(real_df, synthetic_df, col_types['categorical_columns'])

        assert 0.0 <= retention <= 1.0


class TestMissingCategoryRatio:
    """Test missing category ratio metric."""

    def test_missing_categories_none(self):
        """Test when no categories are missing."""
        real_df = pd.DataFrame({'cat': ['A', 'B', 'C']})
        synthetic_df = pd.DataFrame({'cat': ['A', 'B', 'C']})

        missing = compute_missing_category_ratio(real_df, synthetic_df, ['cat'])
        assert missing == 0.0

    def test_missing_categories_all(self):
        """Test when all categories are missing."""
        real_df = pd.DataFrame({'cat': ['A', 'B', 'C']})
        synthetic_df = pd.DataFrame({'cat': ['X', 'Y', 'Z']})

        missing = compute_missing_category_ratio(real_df, synthetic_df, ['cat'])
        assert missing == 1.0

    def test_missing_categories_partial(self):
        """Test when some categories are missing."""
        real_df = pd.DataFrame({'cat': ['A', 'B', 'C', 'D']})
        synthetic_df = pd.DataFrame({'cat': ['A', 'B']})  # Missing C, D

        missing = compute_missing_category_ratio(real_df, synthetic_df, ['cat'])
        assert missing == 0.5  # 2 out of 4 missing

    def test_missing_categories_adult_dataset(self, adult_data):
        """Test on Adult dataset."""
        real_df, synthetic_df, col_types = adult_data
        missing = compute_missing_category_ratio(real_df, synthetic_df, col_types['categorical_columns'])

        assert 0.0 <= missing <= 1.0
        assert missing < 0.3  # Adult train/test should have low missing rate


class TestMissingnessDelta:
    """Test missingness delta metric."""

    def test_missingness_identical(self):
        """Test when missingness is identical."""
        real_df = pd.DataFrame({'a': [1, 2, np.nan], 'b': [4, 5, 6]})
        synthetic_df = pd.DataFrame({'a': [7, 8, np.nan], 'b': [10, 11, 12]})

        delta = compute_missingness_delta(real_df, synthetic_df)
        assert delta == 0.0

    def test_missingness_different(self):
        """Test when missingness is different."""
        real_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})  # 0% null
        synthetic_df = pd.DataFrame({'a': [np.nan, np.nan, np.nan], 'b': [7, 8, 9]})  # 50% null

        delta = compute_missingness_delta(real_df, synthetic_df)
        assert delta == 0.5

    def test_missingness_adult_dataset(self, adult_data):
        """Test on Adult dataset."""
        real_df, synthetic_df, _ = adult_data
        delta = compute_missingness_delta(real_df, synthetic_df)

        assert delta >= 0.0
        assert np.isfinite(delta)


class TestCoverageMetricsIntegration:
    """Test the main compute_coverage_metrics function."""

    def test_compute_all_metrics(self, adult_data):
        """Test computing all coverage metrics at once."""
        real_df, synthetic_df, col_types = adult_data

        metrics = compute_coverage_metrics(
            real_df,
            synthetic_df,
            col_types['categorical_columns']
        )

        # Check all expected metrics are present
        assert 'coverage_uniqueness_ratio' in metrics
        assert 'coverage_rare_category_retention' in metrics
        assert 'coverage_missing_category_ratio' in metrics
        assert 'coverage_missingness_delta' in metrics

        # Check all values are valid
        for key, value in metrics.items():
            assert np.isfinite(value)
            assert value >= 0.0

            # Ratios should be between 0 and 1
            if 'ratio' in key or 'retention' in key or 'delta' in key:
                assert value <= 1.0
