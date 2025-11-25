from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Tuple, Dict, List

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        path: Path to the CSV file

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def detect_column_types(real_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Automatically detect numerical and categorical columns in a DataFrame.

    Args:
        real_df: Input DataFrame

    Returns:
        Dictionary with keys:
            - 'numerical_columns': List of numerical column names
            - 'categorical_columns': List of categorical column names
    """
    numerical_columns = real_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_columns = real_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    return {
        'numerical_columns': numerical_columns,
        'categorical_columns': categorical_columns
    }


def load_real_data(path: str) -> pd.DataFrame:
    """Load the real dataset for evaluation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Real data CSV not found: {path}")
    return pd.read_csv(path)


def iter_synthetic_frames(input_path: str) -> Generator[Tuple[str, pd.DataFrame], None, None]:
    """Yield (path, DataFrame) pairs for every synthetic CSV under ``input_path``."""
    p = Path(input_path)
    if p.is_file():
        yield str(p), pd.read_csv(p)
        return

    if not p.is_dir():
        raise FileNotFoundError(f"Synthetic input path must be a CSV or directory: {input_path}")

    for csv_path in sorted(p.rglob("*.csv")):
        yield str(csv_path), pd.read_csv(csv_path)
