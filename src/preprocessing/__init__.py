"""Preprocessing module for dataset preparation."""

from src.preprocessing.common import (
    create_splits,
    plot_target_distribution,
    save_processed_data,
)
from src.preprocessing.pipelines import (
    preprocess_ames_housing,
    preprocess_auto_mpg,
    preprocess_california_housing,
    preprocess_simple,
)

__all__ = [
    "create_splits",
    "plot_target_distribution",
    "preprocess_ames_housing",
    "preprocess_auto_mpg",
    "preprocess_california_housing",
    "preprocess_simple",
    "save_processed_data",
]
