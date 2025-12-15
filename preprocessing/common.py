"""Common preprocessing utilities for all datasets."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def create_splits(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
    stratify_col: str | None = None,
) -> pd.DataFrame:
    """
    Create train/val/test splits and add a 'split' column to the dataframe.

    Args:
        df: Input dataframe
        train_size: Proportion for training set (default 0.7)
        val_size: Proportion for validation set (default 0.2)
        test_size: Proportion for test set (default 0.1)
        random_state: Random seed for reproducibility
        stratify_col: Column name to use for stratification (optional)

    Returns:
        DataFrame with added 'split' column containing 'train', 'val', or 'test'
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"

    df = df.copy()

    # Stratification setup
    stratify = df[stratify_col] if stratify_col else None

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Second split: val vs test from remaining data
    val_proportion = val_size / (val_size + test_size)
    stratify_temp = temp_df[stratify_col] if stratify_col else None

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_proportion,
        random_state=random_state,
        stratify=stratify_temp,
    )

    # Add split labels
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Combine back together
    result = pd.concat([train_df, val_df, test_df], axis=0)

    return result


def save_processed_data(df: pd.DataFrame, dataset_name: str, output_dir: Path | str) -> None:
    """
    Save processed dataframe to CSV.

    Args:
        df: Processed dataframe with 'split' column
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
