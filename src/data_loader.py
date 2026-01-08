"""Data loading utilities for XGBoost training."""

import numpy as np
import pandas as pd

from src import get_logger

logger = get_logger(__name__)


def load_and_prepare_data(
    dataset_path: str, target_col: str
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, list, list[str]]:
    """
    Load dataset and prepare train/test splits.

    Args:
        dataset_path: Path to CSV file with 'fold' column
        target_col: Name of target column

    Returns:
        Tuple of (X_train_val, y_train_val, X_test, y_test, cv_splits, feature_cols)
    """
    logger.info("Loading data from %s", dataset_path)
    df = pd.read_csv(dataset_path)

    # Identify feature columns
    feature_cols = [c for c in df.columns if c not in [target_col, "fold"]]
    categorical_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()

    # Convert categorical columns
    for col in categorical_cols:
        df[col] = df[col].astype("category")
        logger.info("Categorical: %s (%d unique)", col, df[col].nunique())

    # Split by fold
    train_val_df = df[df["fold"] != "test"].copy()
    test_df = df[df["fold"] == "test"].copy()
    train_val_df["fold"] = train_val_df["fold"].astype(int)

    X_train_val = train_val_df[feature_cols]
    y_train_val = train_val_df[target_col].values
    folds = train_val_df["fold"].values

    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values

    # Build CV splits
    cv_splits = []
    for fold_id in sorted(train_val_df["fold"].unique()):
        train_idx = np.where(folds != fold_id)[0]
        val_idx = np.where(folds == fold_id)[0]
        cv_splits.append((train_idx, val_idx))

    logger.info(
        "Train: %d, Test: %d, Folds: %d, Features: %d",
        len(X_train_val),
        len(X_test),
        len(cv_splits),
        len(feature_cols),
    )

    return X_train_val, y_train_val, X_test, y_test, cv_splits, feature_cols
