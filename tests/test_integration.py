"""
Integration tests for end-to-end workflow.
"""
import pytest
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

from sdeval.data_loader import load_csv, detect_column_types
from sdeval.metrics.statistical import compute_statistical_metrics
from sdeval.metrics.coverage import compute_coverage_metrics
from sdeval.metrics.privacy import compute_privacy_metrics
from sdeval.visualization import create_distribution_plots


# Test data paths
TRAIN_CSV = Path("datasets/adult/train.csv")
TEST_CSV = Path("datasets/adult/test.csv")


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    return tmp_path


class TestEndToEndWorkflow:
    """Test complete evaluation workflow."""

    def test_full_evaluation_pipeline(self, temp_output_dir):
        """Test complete pipeline: load data, compute metrics, visualize."""
        # Step 1: Load data
        real_df = load_csv(TRAIN_CSV)
        synthetic_df = load_csv(TEST_CSV)

        assert len(real_df) > 0
        assert len(synthetic_df) > 0

        # Step 2: Detect column types
        col_types = detect_column_types(real_df)

        assert len(col_types['numerical_columns']) > 0
        assert len(col_types['categorical_columns']) > 0

        # Step 3: Compute statistical metrics
        stats_metrics = compute_statistical_metrics(
            real_df,
            synthetic_df,
            col_types['numerical_columns'],
            col_types['categorical_columns']
        )

        assert 'statistical_alpha_precision' in stats_metrics
        assert 'statistical_beta_recall' in stats_metrics
        assert 'statistical_mean_abs_mean_diff' in stats_metrics
        assert 'statistical_mean_abs_std_diff' in stats_metrics
        assert 'statistical_avg_wasserstein' in stats_metrics

        # Step 4: Compute coverage metrics
        coverage_metrics = compute_coverage_metrics(
            real_df,
            synthetic_df,
            col_types['categorical_columns']
        )

        assert 'coverage_uniqueness_ratio' in coverage_metrics
        assert 'coverage_rare_category_retention' in coverage_metrics
        assert 'coverage_missing_category_ratio' in coverage_metrics
        assert 'coverage_missingness_delta' in coverage_metrics

        # Step 5: Compute privacy metrics (using samples for speed)
        real_sample = real_df.sample(n=min(1000, len(real_df)), random_state=42)
        syn_sample = synthetic_df.sample(n=min(500, len(synthetic_df)), random_state=42)

        privacy_metrics = compute_privacy_metrics(
            real_sample,
            syn_sample,
            col_types['numerical_columns']
        )

        assert 'dcr_rate' in privacy_metrics
        assert 'nndr_mean' in privacy_metrics
        assert 'mean_knn_distance' in privacy_metrics

        # Step 6: Create visualizations
        viz_path = temp_output_dir / "distributions.png"
        create_distribution_plots(
            real_df,
            synthetic_df,
            col_types['numerical_columns'][:2],  # Subset for speed
            col_types['categorical_columns'][:2],
            viz_path
        )

        assert viz_path.exists()

        # Step 7: Save combined results as JSON
        results = {
            'metadata': {
                'real_data_rows': len(real_df),
                'synthetic_data_rows': len(synthetic_df),
                'numerical_columns': col_types['numerical_columns'],
                'categorical_columns': col_types['categorical_columns']
            },
            'statistical': stats_metrics,
            'coverage': coverage_metrics,
            'privacy': privacy_metrics
        }

        output_json = temp_output_dir / "evaluation_results.json"
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)

        assert output_json.exists()

        # Verify JSON is valid
        with open(output_json, 'r') as f:
            loaded_results = json.load(f)

        assert 'metadata' in loaded_results
        assert 'statistical' in loaded_results
        assert 'coverage' in loaded_results
        assert 'privacy' in loaded_results

    def test_metrics_are_reasonable(self):
        """Test that metrics produce reasonable values on Adult dataset."""
        # Load data
        real_df = load_csv(TRAIN_CSV)
        synthetic_df = load_csv(TEST_CSV)
        col_types = detect_column_types(real_df)

        # Compute all metrics
        stats = compute_statistical_metrics(
            real_df, synthetic_df,
            col_types['numerical_columns'],
            col_types['categorical_columns']
        )
        coverage = compute_coverage_metrics(
            real_df, synthetic_df,
            col_types['categorical_columns']
        )

        # Test statistical metrics are in reasonable ranges
        assert 0.7 <= stats['statistical_alpha_precision'] <= 1.0, "Alpha should be high for Adult train/test"
        assert 0.7 <= stats['statistical_beta_recall'] <= 1.0, "Beta should be high for Adult train/test"
        assert stats['statistical_mean_abs_mean_diff'] >= 0
        assert stats['statistical_mean_abs_std_diff'] >= 0
        assert stats['statistical_avg_wasserstein'] >= 0

        # Test coverage metrics
        assert 0.9 <= coverage['coverage_uniqueness_ratio'] <= 1.0, "Adult data should be mostly unique"
        assert 0.0 <= coverage['coverage_rare_category_retention'] <= 1.0
        assert 0.0 <= coverage['coverage_missing_category_ratio'] <= 0.3, "Missing categories should be low"
        assert coverage['coverage_missingness_delta'] >= 0

    def test_evaluation_with_empty_dataframes(self):
        """Test that evaluation handles edge cases gracefully."""
        empty_df = load_csv(TRAIN_CSV).head(0)  # Empty DataFrame with schema
        small_df = load_csv(TRAIN_CSV).head(10)

        col_types = detect_column_types(small_df)

        # Should not crash with empty data
        stats = compute_statistical_metrics(
            empty_df, small_df,
            col_types['numerical_columns'],
            col_types['categorical_columns']
        )

        # Should return valid metrics (even if 0 or default values)
        # Returns base 5 metrics + 2 correlation delta metrics = 7 total
        assert isinstance(stats, dict)
        assert len(stats) == 7
