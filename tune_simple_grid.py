"""
Simple grid search over max_depth for XGBoost with Poisson regression.
Compares three approaches: baseline, monotonic constraints, and progressive training.
Fixed hyperparameters: subsample=0.8, colsample_bytree=0.8, others at defaults.
"""

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def plot_learning_curves(
    evals_result: dict,
    title: str,
    output_path: Path,
    phase1_trees: int | None = None,
):
    """Plot train/validation learning curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Auto-detect metric from evals_result
    train_data = evals_result.get("validation_0", {})
    val_data = evals_result.get("validation_1", {})
    metric = list(train_data.keys())[0] if train_data else "metric"

    train_metric = train_data.get(metric, [])
    val_metric = val_data.get(metric, [])

    if train_metric:
        ax.plot(train_metric, label="Train", alpha=0.8)
    if val_metric:
        ax.plot(val_metric, label="Validation", alpha=0.8)

    if phase1_trees is not None:
        ax.axvline(x=phase1_trees, color="red", linestyle="--", alpha=0.7, label="Phase 1 → 2")

    ax.set_xlabel("Boosting Round")
    ax.set_ylabel(metric)
    ax.set_title(f"Learning Curves: {title}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved learning curves: %s", output_path)


def plot_feature_importance(
    model: XGBRegressor,
    feature_names: list,
    title: str,
    output_path: Path,
    importance_type: str = "gain",
):
    """Plot feature importance bar chart."""
    importance = model.get_booster().get_score(importance_type=importance_type)

    # Map feature indices to names
    feat_imp = {}
    for feat, imp in importance.items():
        # XGBoost uses f0, f1, etc. as feature names
        if feat.startswith("f"):
            idx = int(feat[1:])
            feat_imp[feature_names[idx]] = imp
        else:
            feat_imp[feat] = imp

    # Sort by importance
    sorted_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    names, values = zip(*sorted_imp) if sorted_imp else ([], [])

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.3)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(f"Importance ({importance_type})")
    ax.set_title(f"Feature Importance: {title}")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved feature importance: %s", output_path)


def compute_pdp(
    model: XGBRegressor,
    X: pd.DataFrame,
    feature: str,
    grid_size: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute partial dependence for a single feature.

    Returns:
        Tuple of (grid_values, averaged_predictions)
    """
    # Create grid spanning feature range
    feat_values = X[feature].dropna()
    grid_values = np.linspace(feat_values.min(), feat_values.max(), grid_size)

    avg_predictions = []
    for val in grid_values:
        X_modified = X.copy()
        X_modified[feature] = val
        preds = model.predict(X_modified)
        avg_predictions.append(preds.mean())

    return grid_values, np.array(avg_predictions)


def plot_pdp_with_ice(
    model: XGBRegressor,
    X: pd.DataFrame,
    feature: str,
    title: str,
    output_path: Path,
    grid_size: int = 50,
    n_ice_lines: int = 50,
):
    """Plot PDP with ICE lines for individual observations."""
    feat_values = X[feature].dropna()
    grid_values = np.linspace(feat_values.min(), feat_values.max(), grid_size)

    # Sample observations for ICE lines
    if len(X) > n_ice_lines:
        ice_indices = X.sample(n=n_ice_lines, random_state=SEED).index
    else:
        ice_indices = X.index

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot ICE lines (individual predictions)
    for idx in ice_indices:
        ice_preds = []
        for val in grid_values:
            X_single = X.loc[[idx]].copy()
            X_single[feature] = val
            pred = model.predict(X_single)[0]
            ice_preds.append(pred)
        ax.plot(grid_values, ice_preds, color="steelblue", alpha=0.15, linewidth=0.8)

    # Plot average PDP line
    avg_preds = []
    for val in grid_values:
        X_modified = X.copy()
        X_modified[feature] = val
        preds = model.predict(X_modified)
        avg_preds.append(preds.mean())
    ax.plot(grid_values, avg_preds, color="darkblue", linewidth=2.5, label="Average (PDP)")

    ax.set_xlabel(feature)
    ax.set_ylabel("Prediction")
    ax.set_title(f"PDP + ICE: {title} - {feature}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_pdps_for_monotonic(
    model: XGBRegressor,
    X: pd.DataFrame,
    monotonic_features: list,
    title: str,
    diagnostics_dir: Path,
):
    """Plot PDPs with ICE lines for monotonic features only."""
    for col in monotonic_features:
        if col not in X.columns:
            continue
        plot_pdp_with_ice(
            model, X, col, title,
            output_path=diagnostics_dir / f"{title}_pdp_{col}.png",
        )
    logger.info("Saved %d PDP plots to: %s", len(monotonic_features), diagnostics_dir)


def get_feature_importance_dict(
    model: XGBRegressor,
    feature_names: list,
    importance_type: str = "gain",
) -> dict[str, float]:
    """Get feature importance as a normalized dict (sums to 100)."""
    importance = model.get_booster().get_score(importance_type=importance_type)

    # Map feature indices to names
    feat_imp = {}
    for feat, imp in importance.items():
        if feat.startswith("f"):
            idx = int(feat[1:])
            feat_imp[feature_names[idx]] = imp
        else:
            feat_imp[feat] = imp

    # Fill missing features with 0
    for feat in feature_names:
        if feat not in feat_imp:
            feat_imp[feat] = 0.0

    # Normalize to 100%
    total = sum(feat_imp.values())
    if total > 0:
        feat_imp = {k: (v / total) * 100 for k, v in feat_imp.items()}

    return feat_imp


def generate_diagnostics(
    model: XGBRegressor,
    evals_result: dict,
    X_test: pd.DataFrame,
    feature_names: list,
    monotonic_features: list,
    approach: str,
    diagnostics_dir: Path,
    phase1_trees: int | None = None,
) -> dict[str, float]:
    """
    Generate all diagnostic plots for a model.

    Returns:
        Feature importance dict (normalized to 100%)
    """
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Learning curves
    plot_learning_curves(
        evals_result,
        title=approach,
        output_path=diagnostics_dir / f"{approach}_learning_curves.png",
        phase1_trees=phase1_trees,
    )

    # Feature importance (plot + return dict)
    plot_feature_importance(
        model,
        feature_names,
        title=approach,
        output_path=diagnostics_dir / f"{approach}_feature_importance.png",
    )
    feat_imp = get_feature_importance_dict(model, feature_names)

    # PDPs with ICE lines for monotonic features only
    plot_pdps_for_monotonic(
        model,
        X_test,
        monotonic_features,
        title=approach,
        diagnostics_dir=diagnostics_dir,
    )

    return feat_imp


@dataclass
class CVResult:
    """Result from cross-validation."""

    cv_mape: float
    median_trees: int
    cv_mapes: list
    cv_trees: list


@dataclass
class GridSearchResult:
    """Result from a single grid search iteration."""

    dataset: str
    approach: str
    max_depth: int
    cv_mape: float
    test_mape: float
    n_trees: int
    num_trees_phase1: int | None = None
    phase1_pct: float | None = None
    chosen: bool = False

    def to_dict(self):
        return {
            "dataset": self.dataset,
            "approach": self.approach,
            "max_depth": self.max_depth,
            "cv_mape": self.cv_mape,
            "test_mape": self.test_mape,
            "n_trees": self.n_trees,
            "num_trees_phase1": self.num_trees_phase1,
            "phase1_pct": self.phase1_pct,
            "chosen": self.chosen,
        }


def run_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splits: list,
    params: dict,
    X_transform_fn=None,
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
        cv_mape=np.mean(cv_mapes),
        median_trees=int(np.median(cv_trees)),
        cv_mapes=cv_mapes,
        cv_trees=cv_trees,
    )


def train_final_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    params: dict,
    n_estimators: int,
    xgb_model=None,
    X_transform_fn=None,
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


def train_final_model_with_eval(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    params: dict,
    n_estimators: int,
    xgb_model=None,
) -> tuple[XGBRegressor, float, dict]:
    """
    Train final model with eval tracking for diagnostics.

    Returns:
        Tuple of (trained model, test MAPE, evals_result)
    """
    final_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    final_params["n_estimators"] = n_estimators

    evals_result = {}
    model = XGBRegressor(**final_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        xgb_model=xgb_model,
        verbose=False,
    )
    evals_result = model.evals_result()

    test_preds = model.predict(X_test)
    test_mape = mean_absolute_percentage_error(y_test, test_preds)

    return model, test_mape, evals_result


def run_cv_progressive(
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splits: list,
    params: dict,
    monotonic_features: list,
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

        # Phase 1: Train with monotonic features as NA
        X_tr_phase1 = X_tr.copy()
        for col in monotonic_features:
            X_tr_phase1[col] = X_tr_phase1[col].astype(float)
        X_tr_phase1.loc[:, monotonic_features] = np.nan

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
        cv_mape=np.mean(cv_mapes),
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
    monotonic_features: list,
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
    X_train_phase1 = X_train.copy()
    for col in monotonic_features:
        X_train_phase1[col] = X_train_phase1[col].astype(float)
    X_train_phase1.loc[:, monotonic_features] = np.nan

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


def train_final_progressive_with_eval(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    params: dict,
    monotonic_features: list,
    num_trees_phase1: int,
    total_trees: int,
) -> tuple[XGBRegressor, float, dict]:
    """
    Train final progressive model with eval tracking for diagnostics.

    Returns:
        Tuple of (trained model, test MAPE, evals_result with both phases)
    """
    params_phase1 = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    params_phase1["n_estimators"] = num_trees_phase1

    # Phase 1 with eval tracking
    X_train_phase1 = X_train.copy()
    X_test_phase1 = X_test.copy()
    for col in monotonic_features:
        X_train_phase1[col] = X_train_phase1[col].astype(float)
        X_test_phase1[col] = X_test_phase1[col].astype(float)
    X_train_phase1.loc[:, monotonic_features] = np.nan
    X_test_phase1.loc[:, monotonic_features] = np.nan

    model_phase1 = XGBRegressor(**params_phase1)
    model_phase1.fit(
        X_train_phase1,
        y_train,
        eval_set=[(X_train_phase1, y_train), (X_test_phase1, y_test)],
        verbose=False,
    )
    evals_phase1 = model_phase1.evals_result()

    # Phase 2 with eval tracking
    params_phase2 = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    params_phase2["n_estimators"] = total_trees - num_trees_phase1

    model = XGBRegressor(**params_phase2)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        xgb_model=model_phase1.get_booster(),
        verbose=False,
    )
    evals_phase2 = model.evals_result()

    # Concatenate eval results from both phases
    evals_result = {}
    for key in evals_phase1:
        evals_result[key] = {}
        for metric in evals_phase1[key]:
            evals_result[key][metric] = evals_phase1[key][metric] + evals_phase2[key][metric]

    test_preds = model.predict(X_test)
    test_mape = mean_absolute_percentage_error(y_test, test_preds)

    return model, test_mape, evals_result


def load_and_prepare_data(dataset_path: str, target_col: str):
    """
    Load dataset and prepare train/test splits.

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

    logger.info("Train: %d, Test: %d, Folds: %d, Features: %d",
                len(X_train_val), len(X_test), len(cv_splits), len(feature_cols))

    return X_train_val, y_train_val, X_test, y_test, cv_splits, feature_cols


def run_grid_search(
    dataset_path,
    target_col,
    monotonic_constraints=None,
    max_depths=None,
    learning_rate=0.1,
    objective="count:poisson",
    output_dir="results",
):
    """
    Run grid search over max_depth with three approaches.

    Args:
        dataset_path: Path to CSV file with 'fold' column
        target_col: Name of target column
        monotonic_constraints: Dict mapping feature names to direction (+1 or -1)
        max_depths: List of max_depth values to try
        learning_rate: XGBoost learning rate
        objective: XGBoost objective (count:poisson, reg:squarederror, reg:gamma, etc.)
        output_dir: Directory for output files
    """
    total_start = time.time()

    if monotonic_constraints is None:
        monotonic_constraints = {}
    if max_depths is None:
        max_depths = list(range(1, 11))

    # Extract feature names for convenience
    monotonic_features = list(monotonic_constraints.keys())

    # Create output directory based on dataset name
    dataset_name = Path(dataset_path).stem
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train_val, y_train_val, X_test, y_test, cv_splits, feature_cols = load_and_prepare_data(
        dataset_path, target_col
    )

    logger.info("Monotonic features: %s", monotonic_features)
    logger.info("Objective: %s", objective)
    logger.info("Grid depths: %s", max_depths)

    # Base parameters
    base_params = {
        "n_estimators": 10000,
        "learning_rate": learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": objective,
        "tree_method": "hist",
        "enable_categorical": True,
        "early_stopping_rounds": 50,
        "random_state": SEED,
        "verbosity": 0,
    }

    # Build monotonic constraints tuple for XGBoost (uses direction from dict, 0 if not constrained)
    mono_constraint_tuple = tuple(monotonic_constraints.get(col, 0) for col in feature_cols)

    all_results = []

    # ====================
    # 1. BASELINE
    # ====================
    logger.info("=" * 60)
    logger.info("BASELINE: Grid search over max_depth")
    logger.info("=" * 60)

    for max_depth in max_depths:
        start = time.time()
        params = {**base_params, "max_depth": max_depth}

        cv_result = run_cv(X_train_val, y_train_val, cv_splits, params)
        _, test_mape = train_final_model(
            X_train_val, y_train_val, X_test, y_test, params, cv_result.median_trees
        )

        elapsed = time.time() - start
        logger.info(
            "depth=%2d | CV=%.4f | Test=%.4f | trees=%4d | %.1fs",
            max_depth, cv_result.cv_mape, test_mape, cv_result.median_trees, elapsed
        )

        all_results.append(GridSearchResult(
            dataset=dataset_name,
            approach="baseline",
            max_depth=max_depth,
            cv_mape=cv_result.cv_mape,
            test_mape=test_mape,
            n_trees=cv_result.median_trees,
        ))

    # ====================
    # 2. MONOTONIC
    # ====================
    logger.info("=" * 60)
    logger.info("MONOTONIC: Grid search over max_depth")
    logger.info("=" * 60)

    for max_depth in max_depths:
        start = time.time()
        params = {**base_params, "max_depth": max_depth, "monotone_constraints": mono_constraint_tuple}

        cv_result = run_cv(X_train_val, y_train_val, cv_splits, params)
        _, test_mape = train_final_model(
            X_train_val, y_train_val, X_test, y_test, params, cv_result.median_trees
        )

        elapsed = time.time() - start
        logger.info(
            "depth=%2d | CV=%.4f | Test=%.4f | trees=%4d | %.1fs",
            max_depth, cv_result.cv_mape, test_mape, cv_result.median_trees, elapsed
        )

        all_results.append(GridSearchResult(
            dataset=dataset_name,
            approach="monotonic",
            max_depth=max_depth,
            cv_mape=cv_result.cv_mape,
            test_mape=test_mape,
            n_trees=cv_result.median_trees,
        ))

    # Find best monotonic config
    mono_results = [r for r in all_results if r.approach == "monotonic"]
    best_mono = min(mono_results, key=lambda x: x.cv_mape)
    logger.info("Best monotonic: depth=%d, trees=%d", best_mono.max_depth, best_mono.n_trees)

    # ====================
    # 3. PROGRESSIVE
    # ====================
    logger.info("=" * 60)
    logger.info("PROGRESSIVE: depth=%d, varying warm-start trees", best_mono.max_depth)
    logger.info("=" * 60)

    percentages = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]
    params = {**base_params, "max_depth": best_mono.max_depth, "monotone_constraints": mono_constraint_tuple}

    for pct in percentages:
        start = time.time()
        num_trees_phase1 = max(10, int(best_mono.n_trees * pct))

        cv_result = run_cv_progressive(
            X_train_val, y_train_val, cv_splits, params, monotonic_features, num_trees_phase1
        )
        _, test_mape = train_final_progressive(
            X_train_val, y_train_val, X_test, y_test, params,
            monotonic_features, num_trees_phase1, cv_result.median_trees
        )

        elapsed = time.time() - start
        logger.info(
            "phase1=%4d (%2.0f%%) | CV=%.4f | Test=%.4f | total=%4d | %.1fs",
            num_trees_phase1, pct * 100, cv_result.cv_mape, test_mape, cv_result.median_trees, elapsed
        )

        all_results.append(GridSearchResult(
            dataset=dataset_name,
            approach="progressive",
            max_depth=best_mono.max_depth,
            cv_mape=cv_result.cv_mape,
            test_mape=test_mape,
            n_trees=cv_result.median_trees,
            num_trees_phase1=num_trees_phase1,
            phase1_pct=pct,
        ))

    # Mark best model for each approach as "chosen"
    for approach in ["baseline", "monotonic", "progressive"]:
        approach_results = [r for r in all_results if r.approach == approach]
        if approach_results:
            best = min(approach_results, key=lambda x: x.cv_mape)
            best.chosen = True

    # Save results
    results_df = pd.DataFrame([r.to_dict() for r in all_results])
    results_file = output_path / "grid_results.csv"
    results_df.to_csv(results_file, index=False)

    # Print summary
    total_elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info("SUMMARY (total time: %.1fs)", total_elapsed)
    logger.info("=" * 60)

    baseline_results = [r for r in all_results if r.approach == "baseline"]
    best_baseline = min(baseline_results, key=lambda x: x.cv_mape)

    progressive_results = [r for r in all_results if r.approach == "progressive"]
    best_progressive = min(progressive_results, key=lambda x: x.cv_mape)

    logger.info("BASELINE:     depth=%d, CV=%.4f, Test=%.4f",
                best_baseline.max_depth, best_baseline.cv_mape, best_baseline.test_mape)
    logger.info("MONOTONIC:    depth=%d, CV=%.4f, Test=%.4f",
                best_mono.max_depth, best_mono.cv_mape, best_mono.test_mape)
    logger.info("PROGRESSIVE:  depth=%d, phase1=%d (%.0f%%), CV=%.4f, Test=%.4f",
                best_progressive.max_depth, best_progressive.num_trees_phase1,
                best_progressive.phase1_pct * 100, best_progressive.cv_mape, best_progressive.test_mape)

    delta_mono = best_mono.test_mape - best_baseline.test_mape
    delta_prog = best_progressive.test_mape - best_baseline.test_mape
    logger.info("vs Baseline:  Monotonic %+.4f (%+.1f%%), Progressive %+.4f (%+.1f%%)",
                delta_mono, 100 * delta_mono / best_baseline.test_mape,
                delta_prog, 100 * delta_prog / best_baseline.test_mape)

    logger.info("Results saved to: %s", results_file)

    # ====================
    # DIAGNOSTICS
    # ====================
    logger.info("=" * 60)
    logger.info("GENERATING DIAGNOSTICS for best models")
    logger.info("=" * 60)

    diagnostics_dir = output_path / "diagnostics"
    all_importances = {}

    # Best baseline model
    logger.info("Training best baseline for diagnostics...")
    baseline_params = {**base_params, "max_depth": best_baseline.max_depth}
    model_baseline, _, evals_baseline = train_final_model_with_eval(
        X_train_val, y_train_val, X_test, y_test, baseline_params, best_baseline.n_trees
    )
    all_importances["baseline"] = generate_diagnostics(
        model_baseline, evals_baseline, X_test, feature_cols, monotonic_features,
        "baseline", diagnostics_dir
    )

    # Best monotonic model
    logger.info("Training best monotonic for diagnostics...")
    monotonic_params = {**base_params, "max_depth": best_mono.max_depth, "monotone_constraints": mono_constraint_tuple}
    model_monotonic, _, evals_monotonic = train_final_model_with_eval(
        X_train_val, y_train_val, X_test, y_test, monotonic_params, best_mono.n_trees
    )
    all_importances["monotonic"] = generate_diagnostics(
        model_monotonic, evals_monotonic, X_test, feature_cols, monotonic_features,
        "monotonic", diagnostics_dir
    )

    # Best progressive model
    logger.info("Training best progressive for diagnostics...")
    progressive_params = {**base_params, "max_depth": best_progressive.max_depth, "monotone_constraints": mono_constraint_tuple}
    model_progressive, _, evals_progressive = train_final_progressive_with_eval(
        X_train_val, y_train_val, X_test, y_test, progressive_params,
        monotonic_features, best_progressive.num_trees_phase1, best_progressive.n_trees
    )
    all_importances["progressive"] = generate_diagnostics(
        model_progressive, evals_progressive, X_test, feature_cols, monotonic_features,
        "progressive", diagnostics_dir,
        phase1_trees=best_progressive.num_trees_phase1,
    )

    # Save feature importances to CSV (features in rows)
    imp_df = pd.DataFrame(all_importances)
    imp_df.index.name = "feature"
    imp_df = imp_df.reset_index()
    imp_file = diagnostics_dir / "feature_importances.csv"
    imp_df.to_csv(imp_file, index=False, float_format="%.2f")
    logger.info("Saved feature importances: %s", imp_file)

    # Create grouped bar chart for feature importances
    fig, ax = plt.subplots(figsize=(12, max(6, len(feature_cols) * 0.4)))
    x = np.arange(len(feature_cols))
    width = 0.25

    # Sort features by baseline importance for better visualization
    sorted_features = sorted(feature_cols, key=lambda f: all_importances["baseline"].get(f, 0), reverse=True)

    for i, (approach, color) in enumerate([("baseline", "steelblue"), ("monotonic", "darkorange"), ("progressive", "green")]):
        values = [all_importances[approach].get(f, 0) for f in sorted_features]
        ax.barh(x + i * width, values, width, label=approach.capitalize(), color=color, alpha=0.8)

    ax.set_yticks(x + width)
    ax.set_yticklabels(sorted_features)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (%)")
    ax.set_title("Feature Importance Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    chart_file = diagnostics_dir / "feature_importance_comparison.png"
    plt.savefig(chart_file, dpi=150)
    plt.close()
    logger.info("Saved importance comparison chart: %s", chart_file)

    logger.info("All outputs saved to: %s", output_path)

    return results_df


def parse_monotonic_constraints(mono_args: list[str]) -> dict[str, int]:
    """
    Parse monotonic constraint arguments.

    Args:
        mono_args: List of "feature:direction" strings, e.g., ["weight:-1", "age:+1"]
                   Direction can be +1, 1, -1. Defaults to +1 if omitted.

    Returns:
        Dict mapping feature name to constraint direction (+1 or -1)
    """
    constraints = {}
    for arg in mono_args:
        if ":" in arg:
            feat, direction = arg.rsplit(":", 1)
            direction = int(direction)
        else:
            feat = arg
            direction = 1
        if direction not in (-1, 1):
            raise ValueError(f"Invalid monotonic direction for {feat}: {direction}. Must be +1 or -1.")
        constraints[feat] = direction
    return constraints


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost grid search over max_depth")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset CSV file")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--monotonic", type=str, nargs="*", default=[],
                        help="Monotonic features as 'feature:direction' (e.g., 'weight:-1', 'age:+1'). "
                             "Direction defaults to +1 if omitted.")
    parser.add_argument("--depths", type=int, nargs="*", default=list(range(1, 11)), help="Max depths to try (default: 1-10)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--objective", type=str, default="count:poisson",
                        help="XGBoost objective (count:poisson, reg:squarederror, reg:gamma)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")

    args = parser.parse_args()

    mono_constraints = parse_monotonic_constraints(args.monotonic)

    run_grid_search(
        dataset_path=args.dataset,
        target_col=args.target,
        monotonic_constraints=mono_constraints,
        max_depths=args.depths,
        learning_rate=args.lr,
        objective=args.objective,
        output_dir=args.output_dir,
    )
