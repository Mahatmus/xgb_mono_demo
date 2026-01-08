"""Baseline training functions for XGBoost."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor  # type: ignore[import-untyped]


def train_final_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    params: dict,
    n_estimators: int,
    xgb_model=None,
    X_transform_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> tuple[XGBRegressor, float]:
    """
    Train final model on full training data and evaluate on test set.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        params: XGBoost parameters
        n_estimators: Number of trees to train
        xgb_model: Optional pre-trained booster for warm-start
        X_transform_fn: Optional function to transform X_train

    Returns:
        Tuple of (trained model, test MAPE)
    """
    final_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    final_params["n_estimators"] = n_estimators

    X_fit = X_transform_fn(X_train) if X_transform_fn else X_train

    model = XGBRegressor(**final_params)
    if xgb_model:
        model.fit(X_fit, y_train, xgb_model=xgb_model, verbose=False)
    else:
        model.fit(X_fit, y_train, verbose=False)

    test_preds = model.predict(X_test)
    test_mape = mean_absolute_percentage_error(y_test, test_preds)

    return model, test_mape
