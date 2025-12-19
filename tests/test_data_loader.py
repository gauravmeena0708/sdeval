"""
Test data loader functionality using Adult dataset.
"""
import pytest
import pandas as pd
from pathlib import Path
from sdeval.data_loader import load_csv, detect_column_types


# Test data paths
TRAIN_CSV = Path("datasets/adult/train.csv")
TEST_CSV = Path("datasets/adult/test.csv")


class TestDataLoader:
    """Test CSV loading functionality."""

    def test_load_csv_success(self):
        """Test loading a valid CSV file."""
        df = load_csv(TRAIN_CSV)

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) > 0

    def test_load_csv_file_not_found(self):
        """Test loading a non-existent CSV file."""
        with pytest.raises(FileNotFoundError):
            load_csv(Path("nonexistent.csv"))

    def test_load_both_train_and_test(self):
        """Test loading both train and test CSV files."""
        train_df = load_csv(TRAIN_CSV)
        test_df = load_csv(TEST_CSV)

        assert len(train_df) > 0
        assert len(test_df) > 0
        # Both should have same columns
        assert list(train_df.columns) == list(test_df.columns)


class TestColumnTypeDetection:
    """Test automatic column type detection."""

    def test_detect_column_types_adult_dataset(self):
        """Test column type detection on Adult dataset."""
        df = load_csv(TRAIN_CSV)
        col_types = detect_column_types(df)

        # Check structure
        assert "numerical_columns" in col_types
        assert "categorical_columns" in col_types

        numerical = col_types["numerical_columns"]
        categorical = col_types["categorical_columns"]

        # Adult dataset should have both types
        assert len(numerical) > 0
        assert len(categorical) > 0

        # Total should equal all columns
        assert len(numerical) + len(categorical) == len(df.columns)

    def test_numerical_columns_are_numeric_dtype(self):
        """Test that detected numerical columns actually have numeric dtypes."""
        df = load_csv(TRAIN_CSV)
        col_types = detect_column_types(df)

        for col in col_types["numerical_columns"]:
            assert pd.api.types.is_numeric_dtype(df[col])

    def test_categorical_columns_are_object_dtype(self):
        """Test that detected categorical columns have object/categorical dtypes."""
        df = load_csv(TRAIN_CSV)
        col_types = detect_column_types(df)

        for col in col_types["categorical_columns"]:
            assert pd.api.types.is_object_dtype(df[col]) or \
                   pd.api.types.is_categorical_dtype(df[col])

    def test_no_columns_missed(self):
        """Test that all columns are classified as either numerical or categorical."""
        df = load_csv(TRAIN_CSV)
        col_types = detect_column_types(df)

        all_classified = set(col_types["numerical_columns"]) | set(col_types["categorical_columns"])
        all_columns = set(df.columns)

        assert all_classified == all_columns
