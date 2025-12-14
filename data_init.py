"""
Download datasets for XGBoost monotonic constraints demonstration.

Setup:
1. uv sync
2. Copy .env.template to .env and add your Kaggle API token
3. Accept competition rules at kaggle.com/competitions/house-prices-advanced-regression-techniques
"""

import os
import zipfile
import shutil
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

if not os.getenv('KAGGLE_API_TOKEN'):
    raise SystemExit("KAGGLE_API_TOKEN not found. Copy .env.template to .env and add your token.")

from kaggle.api.kaggle_api_extended import KaggleApi


def download_california_housing(api):
    """Download California Housing dataset from Kaggle."""
    print("Downloading California Housing...")
    api.dataset_download_files('camnugent/california-housing-prices', path='data', unzip=True)
    if os.path.exists('data/housing.csv'):
        os.rename('data/housing.csv', 'data/california_housing.csv')


def download_diamonds(api):
    """Download Diamonds dataset from Kaggle."""
    print("Downloading Diamonds...")
    api.dataset_download_files('shivam2503/diamonds', path='data', unzip=True)


def download_auto_mpg():
    """Download Auto MPG dataset from UCI."""
    print("Downloading Auto MPG...")
    url = 'https://archive.ics.uci.edu/static/public/9/auto+mpg.zip'
    r = requests.get(url)
    with open('data/auto_mpg.zip', 'wb') as f:
        f.write(r.content)
    with zipfile.ZipFile('data/auto_mpg.zip', 'r') as z:
        z.extractall('data/auto_mpg_tmp')
    df = pd.read_csv('data/auto_mpg_tmp/auto-mpg.data', sep=r'\s+',
                     names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                            'acceleration', 'model_year', 'origin', 'car_name'])
    df.to_csv('data/auto_mpg.csv', index=False)
    os.remove('data/auto_mpg.zip')
    shutil.rmtree('data/auto_mpg_tmp')


def download_insurance(api):
    """Download Medical Insurance dataset from Kaggle."""
    print("Downloading Medical Insurance...")
    api.dataset_download_files('mirichoi0218/insurance', path='data', unzip=True)


def download_concrete():
    """Download Concrete Strength dataset from UCI."""
    print("Downloading Concrete Strength...")
    url = 'https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip'
    r = requests.get(url)
    with open('data/concrete.zip', 'wb') as f:
        f.write(r.content)
    with zipfile.ZipFile('data/concrete.zip', 'r') as z:
        z.extractall('data/concrete_tmp')
    df = pd.read_excel('data/concrete_tmp/Concrete_Data.xls')
    df.to_csv('data/concrete_strength.csv', index=False)
    os.remove('data/concrete.zip')
    shutil.rmtree('data/concrete_tmp')


def download_ames(api):
    """Download Ames Housing dataset from Kaggle Competition."""
    print("Downloading Ames Housing...")
    try:
        api.competition_download_files('house-prices-advanced-regression-techniques', path='data')
        with zipfile.ZipFile('data/house-prices-advanced-regression-techniques.zip', 'r') as z:
            z.extractall('data')
        os.remove('data/house-prices-advanced-regression-techniques.zip')
        os.rename('data/train.csv', 'data/ames_housing.csv')
        # Remove extra competition files we don't need
        for f in ['data/test.csv', 'data/sample_submission.csv', 'data/data_description.txt']:
            if os.path.exists(f):
                os.remove(f)
    except Exception as e:
        print(f"Failed to download Ames Housing: {e}")
        print("Note: You must accept competition rules at:")
        print("https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques")


def download_energy():
    """Download Energy Efficiency dataset from UCI."""
    print("Downloading Energy Efficiency...")
    url = 'https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip'
    r = requests.get(url)
    with open('data/energy.zip', 'wb') as f:
        f.write(r.content)
    with zipfile.ZipFile('data/energy.zip', 'r') as z:
        z.extractall('data/energy_tmp')
    df = pd.read_excel('data/energy_tmp/ENB2012_data.xlsx')
    df.to_csv('data/energy_efficiency.csv', index=False)
    os.remove('data/energy.zip')
    shutil.rmtree('data/energy_tmp')


if __name__ == '__main__':
    # Initialize
    os.makedirs('data', exist_ok=True)
    api = KaggleApi()

    # Download all datasets
    download_california_housing(api)
    download_diamonds(api)
    download_auto_mpg()
    download_insurance(api)
    download_concrete()
    download_ames(api)
    download_energy()

    print("\nAll datasets downloaded to data/ directory")
