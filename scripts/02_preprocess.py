#!/usr/bin/env python3
"""
Preprocess all datasets for the XGBoost progressive training experiments.

This script applies dataset-specific preprocessing and creates train/test splits.
Run this after 01_download_data.py.
"""

from src import setup_logging
from src.preprocessing import (
    preprocess_ames_housing,
    preprocess_auto_mpg,
    preprocess_california_housing,
    preprocess_simple,
)

# Configure logging
logger = setup_logging()


def main() -> None:
    """Run all preprocessing pipelines."""
    logger.info("Preprocessing datasets...")
    logger.info("=" * 50)

    # Simple datasets (no variable transformations needed)
    logger.info("\n[1/5] California Housing")
    preprocess_california_housing()

    logger.info("\n[2/5] Medical Insurance")
    preprocess_simple("medical_insurance", "insurance.csv", "charges")

    logger.info("\n[3/5] Concrete Strength")
    preprocess_simple(
        "concrete_strength",
        "concrete_strength.csv",
        "Concrete compressive strength(MPa, megapascals) ",
    )

    # Datasets with variable transformations
    logger.info("\n[4/5] Auto MPG")
    preprocess_auto_mpg()

    logger.info("\n[5/5] Ames Housing")
    preprocess_ames_housing()

    logger.info("\n" + "=" * 50)
    logger.info("All datasets preprocessed and saved to data_processed/")


if __name__ == "__main__":
    main()
