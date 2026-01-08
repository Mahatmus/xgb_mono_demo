"""Training module for XGBoost progressive training."""

from dataclasses import dataclass


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

    def to_dict(self) -> dict:
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


# Re-export training functions
from src.training.baseline import train_final_model
from src.training.cross_validation import run_cv
from src.training.progressive import (
    mask_features_as_constant,
    run_cv_progressive,
    train_final_progressive,
)

__all__ = [
    "CVResult",
    "GridSearchResult",
    "mask_features_as_constant",
    "run_cv",
    "run_cv_progressive",
    "train_final_model",
    "train_final_progressive",
]
