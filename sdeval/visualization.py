"""
Visualization module for generating real vs synthetic comparison plots.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

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
