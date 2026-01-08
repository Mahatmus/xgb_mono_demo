"""XGBoost Progressive Training library."""

import logging

# Reproducibility
SEED = 42

# Progressive training phase 1 percentages
PHASE1_PERCENTAGES = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]

# Default XGBoost base parameters
DEFAULT_BASE_PARAMS = {
    "n_estimators": 10000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "enable_categorical": True,
    "early_stopping_rounds": 50,
    "verbosity": 0,
}


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure centralized logging for the project.

    Args:
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # Override any existing config
    )
    return logging.getLogger(__name__)


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance for a module."""
    return logging.getLogger(name or __name__)
