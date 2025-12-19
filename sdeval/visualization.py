"""
Visualization module for generating real vs synthetic comparison plots.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def create_distribution_plots(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numerical_columns: List[str],
    categorical_columns: List[str],
    output_path: Path,
    max_categories: int = 20
) -> None:
    """
    Create distribution comparison plots for real vs synthetic data.

    Args:
        real_df: Real dataset
        synthetic_df: Synthetic dataset
        numerical_columns: List of numerical column names
        categorical_columns: List of categorical column names
        output_path: Path to save the plot
        max_categories: Maximum number of categories to display in bar plots
    """
    # Calculate grid size
    total_cols = len(numerical_columns) + len(categorical_columns)
    if total_cols == 0:
        return

    ncols = min(3, total_cols)
    nrows = (total_cols + ncols - 1) // ncols

    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if total_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if total_cols > 1 else axes

    plot_idx = 0

    # Plot numerical columns as density plots
    for col in numerical_columns:
        if col not in real_df.columns or col not in synthetic_df.columns:
            continue

        ax = axes[plot_idx]

        # Get data, drop NaN
        real_data = pd.to_numeric(real_df[col], errors='coerce').dropna()
        syn_data = pd.to_numeric(synthetic_df[col], errors='coerce').dropna()

        if len(real_data) > 0:
            sns.kdeplot(real_data, ax=ax, label='Real', fill=True, alpha=0.5, color='darkblue')
        if len(syn_data) > 0:
            sns.kdeplot(syn_data, ax=ax, label='Synthetic', fill=True, alpha=0.5, color='cyan')

        ax.set_title(f"Real vs. Synthetic Data for column '{col}'", fontsize=10)
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_idx += 1

    # Plot categorical columns as bar plots
    for col in categorical_columns:
        if col not in real_df.columns or col not in synthetic_df.columns:
            continue

        ax = axes[plot_idx]

        # Get value counts
        real_counts = real_df[col].value_counts(normalize=True, dropna=True).head(max_categories)
        syn_counts = synthetic_df[col].value_counts(normalize=True, dropna=True).head(max_categories)

        # Combine categories
        all_categories = list(set(real_counts.index) | set(syn_counts.index))
        all_categories = sorted(all_categories)[:max_categories]

        # Prepare data
        real_vals = [real_counts.get(cat, 0) for cat in all_categories]
        syn_vals = [syn_counts.get(cat, 0) for cat in all_categories]

        # Plot
        x = np.arange(len(all_categories))
        width = 0.35

        ax.bar(x - width/2, real_vals, width, label='Real', color='darkblue', alpha=0.7)
        ax.bar(x + width/2, syn_vals, width, label='Synthetic', color='cyan', alpha=0.7)

        ax.set_title(f"Real vs. Synthetic Data for column '{col}'", fontsize=10)
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(all_categories, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_single_column_plot(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    column: str,
    output_path: Path,
    is_numerical: bool = True
) -> None:
    """
    Create a single comparison plot for one column.

    Args:
        real_df: Real dataset
        synthetic_df: Synthetic dataset
        column: Column name to plot
        output_path: Path to save the plot
        is_numerical: Whether the column is numerical (else categorical)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    if is_numerical:
        # Density plot
        real_data = pd.to_numeric(real_df[column], errors='coerce').dropna()
        syn_data = pd.to_numeric(synthetic_df[column], errors='coerce').dropna()

        if len(real_data) > 0:
            sns.kdeplot(real_data, ax=ax, label='Real', fill=True, alpha=0.5, color='darkblue')
        if len(syn_data) > 0:
            sns.kdeplot(syn_data, ax=ax, label='Synthetic', fill=True, alpha=0.5, color='cyan')

        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
    else:
        # Bar plot
        real_counts = real_df[column].value_counts(normalize=True, dropna=True).head(20)
        syn_counts = synthetic_df[column].value_counts(normalize=True, dropna=True).head(20)

        all_categories = list(set(real_counts.index) | set(syn_counts.index))
        all_categories = sorted(all_categories)[:20]

        real_vals = [real_counts.get(cat, 0) for cat in all_categories]
        syn_vals = [syn_counts.get(cat, 0) for cat in all_categories]

        x = np.arange(len(all_categories))
        width = 0.35

        ax.bar(x - width/2, real_vals, width, label='Real', color='darkblue', alpha=0.7)
        ax.bar(x + width/2, syn_vals, width, label='Synthetic', color='cyan', alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(all_categories, rotation=45, ha='right')
        ax.set_xlabel('Category')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3, axis='y')

    ax.set_title(f"Real vs. Synthetic Data for column '{column}'")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_qq_plots(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numerical_columns: List[str],
    output_path: Path,
) -> None:
    """Create QQ plots comparing distributions of numerical columns."""
    columns = [col for col in numerical_columns if col in real_df.columns and col in synthetic_df.columns]
    if not columns:
        return

    ncols = min(2, len(columns))
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for idx, col in enumerate(columns):
        ax = axes[idx]
        real_vals = pd.to_numeric(real_df[col], errors='coerce').dropna()
        syn_vals = pd.to_numeric(synthetic_df[col], errors='coerce').dropna()
        if real_vals.empty or syn_vals.empty:
            ax.set_visible(False)
            continue
        quantile_count = min(len(real_vals), len(syn_vals), 200)
        quantiles = np.linspace(0.01, 0.99, quantile_count)
        real_quantiles = np.quantile(real_vals, quantiles)
        syn_quantiles = np.quantile(syn_vals, quantiles)
        ax.scatter(real_quantiles, syn_quantiles, s=15, alpha=0.7, color='teal')
        min_val = min(real_quantiles.min(), syn_quantiles.min())
        max_val = max(real_quantiles.max(), syn_quantiles.max())
        ax.plot([min_val, max_val], [min_val, max_val], ls='--', color='gray')
        ax.set_title(f"QQ Plot: {col}")
        ax.set_xlabel("Real Quantiles")
        ax.set_ylabel("Synthetic Quantiles")
        ax.grid(alpha=0.3)

    for idx in range(len(columns), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_correlation_heatmaps(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numerical_columns: List[str],
    output_path: Path,
) -> None:
    """Create side-by-side correlation heatmaps for real vs synthetic data."""
    columns = [col for col in numerical_columns if col in real_df.columns and col in synthetic_df.columns]
    if len(columns) < 2:
        return

    real_corr = real_df[columns].corr().fillna(0)
    syn_corr = synthetic_df[columns].corr().fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(real_corr, ax=axes[0], cmap='YlGnBu', vmin=-1, vmax=1, annot=False)
    axes[0].set_title("Real Correlation")
    sns.heatmap(syn_corr, ax=axes[1], cmap='YlOrRd', vmin=-1, vmax=1, annot=False)
    axes[1].set_title("Synthetic Correlation")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_constraint_violation_chart(violations: Dict[str, float], output_path: Path) -> None:
    """Visualize constraint violations as a bar chart."""
    if not violations:
        return

    labels = list(violations.keys())
    values = [min(1.0, max(0.0, violations[label])) for label in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4))
    bars = ax.bar(labels, values, color='salmon', edgecolor='black')
    ax.axhline(0.0, color='black', linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Violation Rate")
    ax.set_title("Constraint Violation Overview")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha='center', va='bottom', fontsize=8)
    ax.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_visualization_suite(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numerical_columns: List[str],
    categorical_columns: List[str],
    output_dir: Path,
    file_stem: str,
    constraint_details: Optional[List[Dict[str, Any]]] = None,
    stat_metrics: Optional[Dict[str, Any]] = None,
    max_numerical: int = 6,
    max_categorical: int = 6,
) -> List[Path]:
    """
    Generate a collection of diagnostic plots for a synthetic dataset.

    Returns a list of created file paths.
    """
    created: List[Path] = []
    out_dir = Path(output_dir) / file_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    num_cols = [col for col in numerical_columns if col in real_df.columns and col in synthetic_df.columns][:max_numerical]
    cat_cols = [col for col in categorical_columns if col in real_df.columns and col in synthetic_df.columns][:max_categorical]

    if num_cols or cat_cols:
        dist_path = out_dir / "distributions.png"
        create_distribution_plots(real_df, synthetic_df, num_cols, cat_cols, dist_path)
        created.append(dist_path)

    if num_cols:
        qq_path = out_dir / "qq_plots.png"
        create_qq_plots(real_df, synthetic_df, num_cols, qq_path)
        created.append(qq_path)

    if len(num_cols) > 1:
        corr_path = out_dir / "correlations.png"
        create_correlation_heatmaps(real_df, synthetic_df, num_cols, corr_path)
        created.append(corr_path)

    if constraint_details:
        violations: Dict[str, float] = {}
        for idx, detail in enumerate(constraint_details):
            if detail.get("passed"):
                continue
            rule_id = detail.get("rule_id") or detail.get("type") or f"rule_{idx + 1}"
            violations[str(rule_id)] = 1.0
        if violations:
            constraint_path = out_dir / "constraint_violations.png"
            create_constraint_violation_chart(violations, constraint_path)
            created.append(constraint_path)

    if stat_metrics:
        summary_rows = [
            ("Alpha Precision", stat_metrics.get("statistical_alpha_precision")),
            ("Beta Recall", stat_metrics.get("statistical_beta_recall")),
            ("Avg KS", stat_metrics.get("statistical_avg_ks")),
            ("Avg chi-square", stat_metrics.get("statistical_avg_chi2")),
            ("chi-square p", stat_metrics.get("statistical_avg_chi2_pvalue")),
            ("Avg JSD", stat_metrics.get("statistical_avg_jsd")),
            ("Corr Δ Fro", stat_metrics.get("statistical_corr_delta_fro")),
            ("Corr Δ MeanAbs", stat_metrics.get("statistical_corr_delta_mean_abs")),
        ]
        fig, ax = plt.subplots(figsize=(6, max(2.5, 0.35 * len(summary_rows))))
        ax.axis("off")
        table_data = []
        for label, value in summary_rows:
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                display = "-"
            elif isinstance(value, float):
                if abs(value) >= 1000:
                    display = f"{value:.2e}"
                else:
                    display = f"{value:.4f}"
            else:
                display = str(value)
            table_data.append([label, display])
        table = ax.table(cellText=table_data, colLabels=["Metric", "Value"], loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)
        ax.set_title("Statistical Summary", fontsize=12, pad=12)
        summary_path = out_dir / "statistical_summary.png"
        fig.tight_layout()
        fig.savefig(summary_path, dpi=180)
        plt.close(fig)
        created.append(summary_path)

    return created
