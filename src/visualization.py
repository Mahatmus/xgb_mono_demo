"""Visualization and diagnostics functions for XGBoost models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor  # type: ignore[import-untyped]

from src import SEED, get_logger

logger = get_logger(__name__)


def plot_feature_importance(
    model: XGBRegressor,
    feature_names: list[str],
    title: str,
    output_path: Path,
    importance_type: str = "gain",
) -> None:
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
    names, values = zip(*sorted_imp, strict=True) if sorted_imp else ([], [])

    _fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.3)))
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
) -> None:
    """Plot PDP with ICE lines for individual observations."""
    feat_values = X[feature].dropna()
    grid_values = np.linspace(feat_values.min(), feat_values.max(), grid_size)

    # Sample observations for ICE lines
    if len(X) > n_ice_lines:
        ice_indices = X.sample(n=n_ice_lines, random_state=SEED).index
    else:
        ice_indices = X.index

    _fig, ax = plt.subplots(figsize=(8, 5))

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
    monotonic_features: list[str],
    title: str,
    diagnostics_dir: Path,
) -> None:
    """Plot PDPs with ICE lines for monotonic features only."""
    for col in monotonic_features:
        if col not in X.columns:
            continue
        plot_pdp_with_ice(
            model,
            X,
            col,
            title,
            output_path=diagnostics_dir / f"{title}_pdp_{col}.png",
        )
    logger.info("Saved %d PDP plots to: %s", len(monotonic_features), diagnostics_dir)


def get_feature_importance_dict(
    model: XGBRegressor,
    feature_names: list[str],
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
    X_test: pd.DataFrame,
    feature_names: list[str],
    monotonic_features: list[str],
    approach: str,
    diagnostics_dir: Path,
) -> dict[str, float]:
    """
    Generate all diagnostic plots for a model.

    Returns:
        Feature importance dict (normalized to 100%)
    """
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

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
