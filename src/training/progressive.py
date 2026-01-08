"""
Progressive training technique for XGBoost with monotonic constraints.

This module implements the core "progressive training" technique:
1. Phase 1: Train a model WITHOUT using monotonic-constrained features
   (by masking them as constant values)
2. Phase 2: Continue training with full features and monotonic constraints,
   warm-starting from the Phase 1 model

This approach allows the model to first learn patterns from unconstrained
features, then incorporate monotonic constraints without losing flexibility.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor  # type: ignore[import-untyped]

from src.training import CVResult


def mask_features_as_constant(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Mask specified features as constant (first row value), preventing model from using them.

    This is the key technique for Phase 1 of progressive training - by setting
    monotonic features to a constant value, the model cannot learn from them,
    forcing it to use only unconstrained features.

    Args:
        df: Input DataFrame
        features: List of feature names to mask

    Returns:
        DataFrame with specified features set to constant values
    """
    df_masked = df.copy()
    for col in features:
        if col not in df_masked.columns:
            continue
        first_val = df_masked[col].iloc[0]
        if df_masked[col].dtype.name == "category":
            # Preserve categorical dtype and full categories when setting constant
            cats = df_masked[col].cat.categories
            loc = cats.get_loc(first_val)
            if not isinstance(loc, int):
                loc = 0  # Fallback if get_loc returns slice/mask
            df_masked[col] = pd.Categorical.from_codes([loc] * len(df_masked), categories=cats)
        else:
            df_masked[col] = first_val
    return df_masked


def run_cv_progressive(
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    params: dict,
    monotonic_features: list[str],
    num_trees_phase1: int,
) -> CVResult:
    """
    Run cross-validation for progressive training (two-phase approach).

    Args:
        X: Feature DataFrame
        y: Target array
        cv_splits: List of (train_idx, val_idx) tuples
        params: XGBoost parameters
        monotonic_features: Features to mask in phase 1
        num_trees_phase1: Number of trees for phase 1

    Returns:
        CVResult with metrics
    """
    cv_mapes = []
    cv_total_trees = []

    params_phase1 = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    params_phase1["n_estimators"] = num_trees_phase1

    for train_idx, val_idx in cv_splits:
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Phase 1: Train with monotonic features as constant
        X_tr_phase1 = mask_features_as_constant(X_tr, monotonic_features)

        model_phase1 = XGBRegressor(**params_phase1)
        model_phase1.fit(X_tr_phase1, y_tr, verbose=False)

        # Phase 2: Continue with full features
        model = XGBRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            xgb_model=model_phase1.get_booster(),
            verbose=False,
        )

        preds = model.predict(X_val)
        cv_mapes.append(mean_absolute_percentage_error(y_val, preds))
        cv_total_trees.append(num_trees_phase1 + model.best_iteration + 1)

    return CVResult(
        cv_mape=float(np.mean(cv_mapes)),
        median_trees=int(np.median(cv_total_trees)),
        cv_mapes=cv_mapes,
        cv_trees=cv_total_trees,
    )


def train_final_progressive(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    params: dict,
    monotonic_features: list[str],
    num_trees_phase1: int,
    total_trees: int,
) -> tuple[XGBRegressor, float]:
    """
    Train final progressive model and evaluate.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        params: XGBoost parameters
        monotonic_features: Features to mask in phase 1
        num_trees_phase1: Trees for phase 1
        total_trees: Total trees (phase 1 + phase 2)

    Returns:
        Tuple of (trained model, test MAPE)
    """
    params_phase1 = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    params_phase1["n_estimators"] = num_trees_phase1

    # Phase 1
    X_train_phase1 = mask_features_as_constant(X_train, monotonic_features)

    model_phase1 = XGBRegressor(**params_phase1)
    model_phase1.fit(X_train_phase1, y_train, verbose=False)

    # Phase 2
    params_phase2 = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    params_phase2["n_estimators"] = total_trees - num_trees_phase1

    model = XGBRegressor(**params_phase2)
    model.fit(X_train, y_train, xgb_model=model_phase1.get_booster(), verbose=False)

    test_preds = model.predict(X_test)
    test_mape = mean_absolute_percentage_error(y_test, test_preds)

    return model, test_mape
