"""
Grid search orchestration for XGBoost progressive training experiments.

This module implements a comprehensive comparison of three XGBoost modeling approaches:

1. BASELINE: Standard XGBoost without any monotonic constraints
2. MONOTONIC: XGBoost with monotonic constraints on specified features
3. PROGRESSIVE: Two-phase training approach where:
   - Phase 1: Train on non-monotonic features only (monotonic features masked)
   - Phase 2: Continue training with full feature set and monotonic constraints

The progressive approach aims to get the best of both worlds - learning flexible
patterns from unconstrained features while still enforcing domain-required
monotonic relationships in the final model.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import DEFAULT_BASE_PARAMS, PHASE1_PERCENTAGES, SEED, get_logger
from src.data_loader import load_and_prepare_data
from src.training import (
    GridSearchResult,
    run_cv,
    run_cv_progressive,
    train_final_model,
    train_final_progressive,
)
from src.visualization import generate_diagnostics

logger = get_logger(__name__)


def run_grid_search(
    dataset_path: str,
    target_col: str,
    monotonic_constraints: dict[str, int] | None = None,
    max_depths: list[int] | None = None,
    learning_rate: float = 0.1,
    objective: str = "count:poisson",
    output_dir: str = "results",
) -> pd.DataFrame:
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

    Returns:
        DataFrame with all grid search results
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
        **DEFAULT_BASE_PARAMS,
        "learning_rate": learning_rate,
        "objective": objective,
        "random_state": SEED,
    }

    # Build monotonic constraints tuple for XGBoost
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
            max_depth,
            cv_result.cv_mape,
            test_mape,
            cv_result.median_trees,
            elapsed,
        )

        all_results.append(
            GridSearchResult(
                dataset=dataset_name,
                approach="baseline",
                max_depth=max_depth,
                cv_mape=cv_result.cv_mape,
                test_mape=test_mape,
                n_trees=cv_result.median_trees,
            )
        )

    # ====================
    # 2. MONOTONIC
    # ====================
    logger.info("=" * 60)
    logger.info("MONOTONIC: Grid search over max_depth")
    logger.info("=" * 60)

    for max_depth in max_depths:
        start = time.time()
        params = {
            **base_params,
            "max_depth": max_depth,
            "monotone_constraints": mono_constraint_tuple,
        }

        cv_result = run_cv(X_train_val, y_train_val, cv_splits, params)
        _, test_mape = train_final_model(
            X_train_val, y_train_val, X_test, y_test, params, cv_result.median_trees
        )

        elapsed = time.time() - start
        logger.info(
            "depth=%2d | CV=%.4f | Test=%.4f | trees=%4d | %.1fs",
            max_depth,
            cv_result.cv_mape,
            test_mape,
            cv_result.median_trees,
            elapsed,
        )

        all_results.append(
            GridSearchResult(
                dataset=dataset_name,
                approach="monotonic",
                max_depth=max_depth,
                cv_mape=cv_result.cv_mape,
                test_mape=test_mape,
                n_trees=cv_result.median_trees,
            )
        )

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

    params = {
        **base_params,
        "max_depth": best_mono.max_depth,
        "monotone_constraints": mono_constraint_tuple,
    }

    for pct in PHASE1_PERCENTAGES:
        start = time.time()
        num_trees_phase1 = max(5, int(best_mono.n_trees * pct))

        cv_result = run_cv_progressive(
            X_train_val, y_train_val, cv_splits, params, monotonic_features, num_trees_phase1
        )
        _, test_mape = train_final_progressive(
            X_train_val,
            y_train_val,
            X_test,
            y_test,
            params,
            monotonic_features,
            num_trees_phase1,
            cv_result.median_trees,
        )

        elapsed = time.time() - start
        logger.info(
            "phase1=%4d (%2.0f%%) | CV=%.4f | Test=%.4f | total=%4d | %.1fs",
            num_trees_phase1,
            pct * 100,
            cv_result.cv_mape,
            test_mape,
            cv_result.median_trees,
            elapsed,
        )

        all_results.append(
            GridSearchResult(
                dataset=dataset_name,
                approach="progressive",
                max_depth=best_mono.max_depth,
                cv_mape=cv_result.cv_mape,
                test_mape=test_mape,
                n_trees=cv_result.median_trees,
                num_trees_phase1=num_trees_phase1,
                phase1_pct=pct,
            )
        )

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

    logger.info(
        "BASELINE:     depth=%d, CV=%.4f, Test=%.4f",
        best_baseline.max_depth,
        best_baseline.cv_mape,
        best_baseline.test_mape,
    )
    logger.info(
        "MONOTONIC:    depth=%d, CV=%.4f, Test=%.4f",
        best_mono.max_depth,
        best_mono.cv_mape,
        best_mono.test_mape,
    )
    logger.info(
        "PROGRESSIVE:  depth=%d, phase1=%d (%.0f%%), CV=%.4f, Test=%.4f",
        best_progressive.max_depth,
        best_progressive.num_trees_phase1,
        best_progressive.phase1_pct * 100,
        best_progressive.cv_mape,
        best_progressive.test_mape,
    )

    delta_mono = best_mono.test_mape - best_baseline.test_mape
    delta_prog = best_progressive.test_mape - best_baseline.test_mape
    logger.info(
        "vs Baseline:  Monotonic %+.4f (%+.1f%%), Progressive %+.4f (%+.1f%%)",
        delta_mono,
        100 * delta_mono / best_baseline.test_mape,
        delta_prog,
        100 * delta_prog / best_baseline.test_mape,
    )

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
    model_baseline, _ = train_final_model(
        X_train_val, y_train_val, X_test, y_test, baseline_params, best_baseline.n_trees
    )
    all_importances["baseline"] = generate_diagnostics(
        model_baseline, X_test, feature_cols, monotonic_features, "baseline", diagnostics_dir
    )

    # Best monotonic model
    logger.info("Training best monotonic for diagnostics...")
    monotonic_params = {
        **base_params,
        "max_depth": best_mono.max_depth,
        "monotone_constraints": mono_constraint_tuple,
    }
    model_monotonic, _ = train_final_model(
        X_train_val, y_train_val, X_test, y_test, monotonic_params, best_mono.n_trees
    )
    all_importances["monotonic"] = generate_diagnostics(
        model_monotonic, X_test, feature_cols, monotonic_features, "monotonic", diagnostics_dir
    )

    # Best progressive model
    logger.info("Training best progressive for diagnostics...")
    progressive_params = {
        **base_params,
        "max_depth": best_progressive.max_depth,
        "monotone_constraints": mono_constraint_tuple,
    }
    model_progressive, _ = train_final_progressive(
        X_train_val,
        y_train_val,
        X_test,
        y_test,
        progressive_params,
        monotonic_features,
        best_progressive.num_trees_phase1,
        best_progressive.n_trees,
    )
    all_importances["progressive"] = generate_diagnostics(
        model_progressive, X_test, feature_cols, monotonic_features, "progressive", diagnostics_dir
    )

    # Save feature importances to CSV (features in rows)
    imp_df = pd.DataFrame(all_importances)
    imp_df.index.name = "feature"
    imp_df = imp_df.reset_index()
    imp_file = diagnostics_dir / "feature_importances.csv"
    imp_df.to_csv(imp_file, index=False, float_format="%.2f")
    logger.info("Saved feature importances: %s", imp_file)

    # Create grouped bar chart for feature importances
    _fig, ax = plt.subplots(figsize=(12, max(6, len(feature_cols) * 0.4)))
    x = np.arange(len(feature_cols))
    width = 0.25

    # Sort features by baseline importance for better visualization
    sorted_features = sorted(
        feature_cols, key=lambda f: all_importances["baseline"].get(f, 0), reverse=True
    )

    for i, (approach, color) in enumerate(
        [("baseline", "steelblue"), ("monotonic", "darkorange"), ("progressive", "green")]
    ):
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
