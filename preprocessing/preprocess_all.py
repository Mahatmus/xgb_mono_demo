"""Preprocessing pipelines for all datasets."""

from pathlib import Path

import pandas as pd

from preprocessing.common import create_splits, plot_target_distribution, save_processed_data


def preprocess_simple(
    dataset_name: str,
    input_filename: str,
    target_col: str,
    input_dir: Path | str = "data",
    output_dir: Path | str = "data_processed",
) -> None:
    """
    Generic preprocessing for datasets that only need fold-based splits.

    Used for: California Housing, Medical Insurance, Concrete Strength

    Args:
        dataset_name: Name for output file
        input_filename: Input CSV filename
        target_col: Name of the target column for plotting and stratification
        input_dir: Directory containing raw data
        output_dir: Directory to save processed data
    """
    input_path = Path(input_dir) / input_filename
    df = pd.read_csv(input_path)

    # Create stratified folds (6 folds total, fold 6 = test)
    df = create_splits(df, n_folds=6, random_state=42, stratify_col=target_col)

    # Save processed data
    save_processed_data(df, dataset_name, output_dir)

    # Plot target distribution
    plot_target_distribution(df, target_col, dataset_name, output_dir)


def preprocess_auto_mpg(
    input_dir: Path | str = "data",
    output_dir: Path | str = "data_processed",
) -> None:
    """
    Preprocess Auto MPG dataset.

    Preprocessing steps:
    1. Extract manufacturer from car_name
    2. Handle missing horsepower values
    3. Drop car_name column
    4. Create stratified fold-based splits

    Args:
        input_dir: Directory containing raw data
        output_dir: Directory to save processed data
    """
    input_path = Path(input_dir) / "auto_mpg.csv"
    df = pd.read_csv(input_path)

    # Extract manufacturer from car_name (first word)
    df["manufacturer"] = df["car_name"].str.split().str[0].str.lower()

    # Normalize manufacturer names (fix typos and variants)
    manufacturer_map = {
        "chevy": "chevrolet",
        "chevroelt": "chevrolet",
        "toyouta": "toyota",
        "vokswagen": "volkswagen",
        "vw": "volkswagen",
        "maxda": "mazda",
        "hi": "bmw",  # "bmw 2002" appears as "hi" in some entries
    }
    df["manufacturer"] = df["manufacturer"].replace(manufacturer_map)

    # Group rare manufacturers (< 10 samples) into "other"
    manufacturer_counts = df["manufacturer"].value_counts()
    rare_manufacturers = manufacturer_counts[manufacturer_counts < 10].index
    df.loc[df["manufacturer"].isin(rare_manufacturers), "manufacturer"] = "other"

    # Drop car_name
    df = df.drop(columns=["car_name"])

    # Handle missing horsepower (marked as '?' in original data)
    # Convert to numeric, which will turn '?' into NaN
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")

    # Create stratified folds (6 folds total, fold 6 = test), stratified by manufacturer
    df = create_splits(df, n_folds=6, random_state=42, stratify_col="manufacturer")

    # Save processed data
    save_processed_data(df, "auto_mpg", output_dir)

    # Plot target distribution
    plot_target_distribution(df, "mpg", "auto_mpg", output_dir)


def preprocess_ames_housing(
    input_dir: Path | str = "data",
    output_dir: Path | str = "data_processed",
) -> None:
    """
    Preprocess Ames Housing dataset.

    Preprocessing steps:
    1. Select variables according to data_descriptions/ames_housing.md
    2. Create ratio features (relative to GrLivArea)
    3. Encode ordinal variables
    4. Create stratified fold-based splits

    Args:
        input_dir: Directory containing raw data
        output_dir: Directory to save processed data
    """
    input_path = Path(input_dir) / "ames_housing.csv"
    df = pd.read_csv(input_path)

    # Variables to keep (with monotonic constraints)
    keep_vars = ["GrLivArea", "LotArea"]

    # Variables to convert to ratios
    # Note: 2ndFlrSF excluded because 1stFlrSF + 2ndFlrSF = GrLivArea (perfectly correlated)
    ratio_vars = ["1stFlrSF", "TotalBsmtSF", "GarageArea", "WoodDeckSF", "OpenPorchSF"]

    # Unconstrained variables to keep
    unconstrained = [
        "YearBuilt",
        "FullBath",
        "Fireplaces",
        "OverallQual",
        "OverallCond",
        "ExterQual",
        "KitchenQual",
        "BsmtQual",
    ]

    # Create ratio features (as percentages, rounded to 2 decimals)
    for var in ratio_vars:
        if var in df.columns:
            df[f"{var}_ratio"] = (df[var] / df["GrLivArea"] * 100).round(2)

    # Ordinal encoding mappings
    quality_map = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    bsmt_qual_map = {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

    # Apply ordinal encoding
    if "ExterQual" in df.columns:
        df["ExterQual"] = df["ExterQual"].map(quality_map)
    if "KitchenQual" in df.columns:
        df["KitchenQual"] = df["KitchenQual"].map(quality_map)
    if "BsmtQual" in df.columns:
        df["BsmtQual"] = df["BsmtQual"].fillna("NA").map(bsmt_qual_map)

    # Select final columns
    ratio_cols = [f"{var}_ratio" for var in ratio_vars if var in df.columns]
    final_cols = keep_vars + ratio_cols + unconstrained + ["SalePrice"]
    df = df[final_cols]

    # Create stratified folds (6 folds total, fold 6 = test)
    df = create_splits(df, n_folds=6, random_state=42, stratify_col="SalePrice")

    # Save processed data
    save_processed_data(df, "ames_housing", output_dir)

    # Plot target distribution
    plot_target_distribution(df, "SalePrice", "ames_housing", output_dir)


def preprocess_california_housing(
    input_dir: Path | str = "data",
    output_dir: Path | str = "data_processed",
) -> None:
    """
    Preprocess California Housing dataset.

    Preprocessing steps:
    1. Remove houses with median_house_value > 500,000 (censored values)
    2. Create stratified fold-based splits

    Args:
        input_dir: Directory containing raw data
        output_dir: Directory to save processed data
    """
    input_path = Path(input_dir) / "california_housing.csv"
    df = pd.read_csv(input_path)

    # Remove censored values (houses capped at $500k+)
    df = df[df["median_house_value"] <= 500000]

    # Create stratified folds (6 folds total, fold 6 = test)
    df = create_splits(df, n_folds=6, random_state=42, stratify_col="median_house_value")

    # Save processed data
    save_processed_data(df, "california_housing", output_dir)

    # Plot target distribution
    plot_target_distribution(df, "median_house_value", "california_housing", output_dir)


if __name__ == "__main__":
    # Simple datasets (no variable transformations needed)
    preprocess_california_housing()
    preprocess_simple("medical_insurance", "insurance.csv", "charges")
    preprocess_simple(
        "concrete_strength",
        "concrete_strength.csv",
        "Concrete compressive strength(MPa, megapascals) ",
    )

    # Datasets with variable transformations
    preprocess_auto_mpg()
    preprocess_ames_housing()
