#!/usr/bin/env python3
"""
Download datasets for XGBoost monotonic constraints demonstration.

Setup:
1. uv sync
2. Copy .env.template to .env and add your Kaggle API token
3. Accept competition rules at kaggle.com/competitions/house-prices-advanced-regression-techniques
"""

import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src import setup_logging

# Configure logging
logger = setup_logging()

# Load environment variables BEFORE importing Kaggle API
load_dotenv()

# Set KAGGLE_KEY from KAGGLE_API_TOKEN if available
if "KAGGLE_API_TOKEN" in os.environ and "KAGGLE_KEY" not in os.environ:
    os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]
    os.environ["KAGGLE_USERNAME"] = "unused"  # Placeholder - not validated with new tokens

# NOW import Kaggle API after environment is configured
# Note: kaggle module auto-authenticates on import and deletes KAGGLE_API_TOKEN
# So we use the pre-authenticated global api instance
import kaggle  # noqa: E402


def download_california_housing(api: kaggle.KaggleApi) -> None:
    """Download California Housing dataset from Kaggle."""
    logger.info("Downloading California Housing...")
    api.dataset_download_files("camnugent/california-housing-prices", path="data", unzip=True)
    housing_path = Path("data/housing.csv")
    if housing_path.exists():
        housing_path.rename("data/california_housing.csv")


def download_auto_mpg() -> None:
    """Download Auto MPG dataset from UCI."""
    logger.info("Downloading Auto MPG...")
    url = "https://archive.ics.uci.edu/static/public/9/auto+mpg.zip"
    r = requests.get(url)
    zip_path = Path("data/auto_mpg.zip")
    zip_path.write_bytes(r.content)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("data/auto_mpg_tmp")
    df = pd.read_csv(
        "data/auto_mpg_tmp/auto-mpg.data",
        sep=r"\s+",
        names=[
            "mpg",
            "cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "model_year",
            "origin",
            "car_name",
        ],
    )
    df.to_csv("data/auto_mpg.csv", index=False)
    zip_path.unlink()
    shutil.rmtree("data/auto_mpg_tmp")


def download_insurance(api: kaggle.KaggleApi) -> None:
    """Download Medical Insurance dataset from Kaggle."""
    logger.info("Downloading Medical Insurance...")
    api.dataset_download_files("mirichoi0218/insurance", path="data", unzip=True)


def download_concrete() -> None:
    """Download Concrete Strength dataset from UCI."""
    logger.info("Downloading Concrete Strength...")
    url = "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"
    r = requests.get(url)
    zip_path = Path("data/concrete.zip")
    zip_path.write_bytes(r.content)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("data/concrete_tmp")
    df = pd.read_excel("data/concrete_tmp/Concrete_Data.xls")
    df.to_csv("data/concrete_strength.csv", index=False)
    zip_path.unlink()
    shutil.rmtree("data/concrete_tmp")


def download_ames(api: kaggle.KaggleApi) -> None:
    """Download Ames Housing dataset from Kaggle Competition."""
    logger.info("Downloading Ames Housing...")
    try:
        api.competition_download_files("house-prices-advanced-regression-techniques", path="data")
        zip_path = Path("data/house-prices-advanced-regression-techniques.zip")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("data")
        zip_path.unlink()
        Path("data/train.csv").rename("data/ames_housing.csv")
        # Remove extra competition files we don't need
        for f in ["data/test.csv", "data/sample_submission.csv", "data/data_description.txt"]:
            file_path = Path(f)
            if file_path.exists():
                file_path.unlink()
    except Exception as e:
        logger.error("Failed to download Ames Housing: %s", e)
        logger.info("Note: You must accept competition rules at:")
        logger.info(
            "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques"
        )


if __name__ == "__main__":
    # Initialize
    Path("data").mkdir(exist_ok=True)
    # Use pre-authenticated global api (authenticated during import)
    api = kaggle.api

    # Download all datasets
    download_california_housing(api)
    download_auto_mpg()
    download_insurance(api)
    download_concrete()
    download_ames(api)

    logger.info("All datasets downloaded to data/ directory")
