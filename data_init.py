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

# Load environment variables BEFORE importing Kaggle API
load_dotenv()

# Set KAGGLE_KEY from KAGGLE_API_TOKEN if available
if "KAGGLE_API_TOKEN" in os.environ and "KAGGLE_KEY" not in os.environ:
    os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]
    os.environ["KAGGLE_USERNAME"] = ""  # Not needed with API token

# NOW import Kaggle API after environment is configured
from kaggle.api.kaggle_api_extended import KaggleApi


def download_california_housing(api: KaggleApi) -> None:
    """Download California Housing dataset from Kaggle."""
    print("Downloading California Housing...")
    api.dataset_download_files("camnugent/california-housing-prices", path="data", unzip=True)
    housing_path = Path("data/housing.csv")
    if housing_path.exists():
        housing_path.rename("data/california_housing.csv")


def download_auto_mpg() -> None:
    """Download Auto MPG dataset from UCI."""
    print("Downloading Auto MPG...")
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


def download_insurance(api: KaggleApi) -> None:
    """Download Medical Insurance dataset from Kaggle."""
    print("Downloading Medical Insurance...")
    api.dataset_download_files("mirichoi0218/insurance", path="data", unzip=True)


def download_concrete() -> None:
    """Download Concrete Strength dataset from UCI."""
    print("Downloading Concrete Strength...")
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


def download_ames(api: KaggleApi) -> None:
    """Download Ames Housing dataset from Kaggle Competition."""
    print("Downloading Ames Housing...")
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
        print(f"Failed to download Ames Housing: {e}")
        print("Note: You must accept competition rules at:")
        print("https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques")


def download_energy() -> None:
    """Download Energy Efficiency dataset from UCI."""
    print("Downloading Energy Efficiency...")
    url = "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"
    r = requests.get(url)
    zip_path = Path("data/energy.zip")
    zip_path.write_bytes(r.content)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("data/energy_tmp")
    df = pd.read_excel("data/energy_tmp/ENB2012_data.xlsx")

    # Rename columns from X1-X8, Y1-Y2 to descriptive names
    df.columns = [
        "Relative Compactness",
        "Surface Area",
        "Wall Area",
        "Roof Area",
        "Overall Height",
        "Orientation",
        "Glazing Area",
        "Glazing Area Distribution",
        "Heating Load",
        "Cooling Load",
    ]

    df.to_csv("data/energy_efficiency.csv", index=False)
    zip_path.unlink()
    shutil.rmtree("data/energy_tmp")


if __name__ == "__main__":
    # Initialize
    Path("data").mkdir(exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    # Download all datasets
    download_california_housing(api)
    download_auto_mpg()
    download_insurance(api)
    download_concrete()
    download_ames(api)
    download_energy()

    print("\nAll datasets downloaded to data/ directory")
