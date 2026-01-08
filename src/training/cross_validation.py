"""Cross-validation functions for XGBoost training."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor  # type: ignore[import-untyped]

from src.training import CVResult


def run_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    params: dict,
    X_transform_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> CVResult:
    """
    Run cross-validation with early stopping.

    Args:
        X: Feature DataFrame
        y: Target array
        cv_splits: List of (train_idx, val_idx) tuples
        params: XGBoost parameters (must include early_stopping_rounds)
        X_transform_fn: Optional function to transform X_train before fitting

    Returns:
        CVResult with metrics
    """
    cv_mapes = []
    cv_trees = []

    for train_idx, val_idx in cv_splits:
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if X_transform_fn:
            X_tr = X_transform_fn(X_tr)

        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        preds = model.predict(X_val)
        cv_mapes.append(mean_absolute_percentage_error(y_val, preds))
        cv_trees.append(model.best_iteration + 1)

    return CVResult(
        cv_mape=float(np.mean(cv_mapes)),
        median_trees=int(np.median(cv_trees)),
        cv_mapes=cv_mapes,
        cv_trees=cv_trees,
    )
