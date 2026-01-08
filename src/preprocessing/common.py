"""Common preprocessing utilities for all datasets."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def create_splits(
    df: pd.DataFrame,
    n_folds: int = 6,
    random_state: int = 42,
    stratify_col: str | None = None,
) -> pd.DataFrame:
    """
    Create K-fold CV splits with last fold as test set and add a 'fold' column.

    Args:
        df: Input dataframe
        n_folds: Total number of folds (default 6, with fold 6 as test)
        random_state: Random seed for reproducibility
        stratify_col: Column name to use for stratification (optional)

    Returns:
        DataFrame with 'fold' column containing 1 to (n_folds-1) or 'test' for last fold
    """
    df = df.copy()

    if stratify_col:
        y = df[stratify_col].to_numpy()

        # Check if stratify column is numeric (continuous) or categorical
        is_numeric = pd.api.types.is_numeric_dtype(df[stratify_col])

        if is_numeric:
            # For continuous targets, create bins for stratification
            min_samples_per_bin = n_folds * 2
            n_bins = max(3, min(10, len(y) // min_samples_per_bin))

            # Use quantile-based binning
            bins = np.percentile(y, np.linspace(0, 100, n_bins + 1))
            bins = np.unique(bins)  # Remove duplicates

            # Ensure we have at least 2 bins
            if len(bins) <= 2:
                median_val = np.median(y)
                y_stratify = (y > median_val).astype(int)
            else:
                y_stratify = np.digitize(y, bins[1:-1])
        else:
            # For categorical variables, use them directly
            y_stratify = y

        # Create stratified K-folds
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_assignments = np.zeros(len(df), dtype=int)

        for fold_idx, (_, val_idx) in enumerate(skf.split(df, y_stratify), 1):
            fold_assignments[val_idx] = fold_idx
    else:
        # Random assignment if no stratification
        indices = np.arange(len(df))
        np.random.seed(random_state)
        np.random.shuffle(indices)
        fold_assignments = (indices % n_folds) + 1

    # Assign folds: 1 to n_folds-1 stay as integers, n_folds becomes 'test'
    df["fold"] = fold_assignments.astype(object)
    df.loc[df["fold"] == n_folds, "fold"] = "test"

    return df


def save_processed_data(df: pd.DataFrame, dataset_name: str, output_dir: Path | str) -> None:
    """
    Save processed dataframe to CSV.

    Args:
        df: Processed dataframe with 'fold' column
        dataset_name: Name of the dataset (used for filename)
        output_dir: Directory to save the processed data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{dataset_name}.csv"
    df.to_csv(output_path, index=False)


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    dataset_name: str,
    output_dir: Path | str,
) -> None:
    """
    Plot and save target variable distribution as PNG.

    Args:
        df: Dataframe containing the target variable
        target_col: Name of the target column
        dataset_name: Name of the dataset (used for filename)
        output_dir: Directory to save the plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(df[target_col], bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel(target_col)
    plt.ylabel("Frequency")
    plt.title(f"{dataset_name}: {target_col} Distribution")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / f"{dataset_name}_target_distribution.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
