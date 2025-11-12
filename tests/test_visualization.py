"""
Test visualization functionality.
"""
import pytest
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from sdeval.data_loader import load_csv, detect_column_types
from sdeval.visualization import create_distribution_plots, create_single_column_plot


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
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    return tmp_path


class TestDistributionPlots:
    """Test distribution plot generation."""

    def test_create_distribution_plots_adult_dataset(self, adult_data, temp_output_dir):
        """Test creating distribution plots with Adult dataset."""
        real_df, synthetic_df, col_types = adult_data

        # Use subset of columns for speed
        num_cols = col_types['numerical_columns'][:2]
        cat_cols = col_types['categorical_columns'][:2]

        output_path = temp_output_dir / "distribution_plots.png"

        create_distribution_plots(
            real_df,
            synthetic_df,
            num_cols,
            cat_cols,
            output_path
        )

        # Check that file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_create_distribution_plots_all_numerical(self, temp_output_dir):
        """Test with only numerical columns."""
        real_df = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5],
            'num2': [10, 20, 30, 40, 50]
        })
        synthetic_df = pd.DataFrame({
            'num1': [1.1, 2.1, 3.1, 4.1, 5.1],
            'num2': [11, 21, 31, 41, 51]
        })

        output_path = temp_output_dir / "numerical_only.png"

        create_distribution_plots(
            real_df,
            synthetic_df,
            ['num1', 'num2'],
            [],
            output_path
        )

        assert output_path.exists()

    def test_create_distribution_plots_all_categorical(self, temp_output_dir):
        """Test with only categorical columns."""
        real_df = pd.DataFrame({
            'cat1': ['A', 'B', 'C', 'A', 'B'],
            'cat2': ['X', 'Y', 'Z', 'X', 'Y']
        })
        synthetic_df = pd.DataFrame({
            'cat1': ['A', 'B', 'C', 'D'],
            'cat2': ['X', 'Y', 'Z', 'W']
        })

        output_path = temp_output_dir / "categorical_only.png"

        create_distribution_plots(
            real_df,
            synthetic_df,
            [],
            ['cat1', 'cat2'],
            output_path
        )

        assert output_path.exists()

    def test_create_distribution_plots_empty_columns(self, temp_output_dir):
        """Test with no columns (should handle gracefully)."""
        real_df = pd.DataFrame({'a': [1, 2, 3]})
        synthetic_df = pd.DataFrame({'a': [4, 5, 6]})

        output_path = temp_output_dir / "empty.png"

        # Should not create file when no columns specified
        create_distribution_plots(
            real_df,
            synthetic_df,
            [],
            [],
            output_path
        )

        # File should not exist for empty input
        assert not output_path.exists()


class TestSingleColumnPlot:
    """Test single column plot generation."""

    def test_create_single_numerical_plot(self, adult_data, temp_output_dir):
        """Test single numerical column plot."""
        real_df, synthetic_df, col_types = adult_data

        if col_types['numerical_columns']:
            col = col_types['numerical_columns'][0]
            output_path = temp_output_dir / f"{col}_plot.png"

            create_single_column_plot(
                real_df,
                synthetic_df,
                col,
                output_path,
                is_numerical=True
            )

            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_create_single_categorical_plot(self, adult_data, temp_output_dir):
        """Test single categorical column plot."""
        real_df, synthetic_df, col_types = adult_data

        if col_types['categorical_columns']:
            col = col_types['categorical_columns'][0]
            output_path = temp_output_dir / f"{col}_plot.png"

            create_single_column_plot(
                real_df,
                synthetic_df,
                col,
                output_path,
                is_numerical=False
            )

            assert output_path.exists()
            assert output_path.stat().st_size > 0
