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
    Generic preprocessing for datasets that only need train/val/test splits.

    Used for: California Housing, Medical Insurance, Concrete Strength

    Args:
        dataset_name: Name for output file
        input_filename: Input CSV filename
        target_col: Name of the target column for plotting
        input_dir: Directory containing raw data
        output_dir: Directory to save processed data
    """
    input_path = Path(input_dir) / input_filename
    df = pd.read_csv(input_path)

    # Create splits (no stratification for continuous targets)
    df = create_splits(df, train_size=0.7, val_size=0.2, test_size=0.1, random_state=42)

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
    4. Create train/val/test splits

    Args:
        input_dir: Directory containing raw data
        output_dir: Directory to save processed data
    """
    input_path = Path(input_dir) / "auto_mpg.csv"
    df = pd.read_csv(input_path)

    # Extract manufacturer from car_name (first word)
    df["manufacturer"] = df["car_name"].str.split().str[0]

    # Drop car_name
    df = df.drop(columns=["car_name"])

    # Handle missing horsepower (marked as '?' in original data)
    # Convert to numeric, which will turn '?' into NaN
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")

    # Create splits (no stratification needed)
    df = create_splits(df, train_size=0.7, val_size=0.2, test_size=0.1, random_state=42)

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
    4. Create train/val/test splits

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

    # Create splits (no stratification needed)
    df = create_splits(df, train_size=0.7, val_size=0.2, test_size=0.1, random_state=42)

    # Save processed data
    save_processed_data(df, "ames_housing", output_dir)

    # Plot target distribution
    plot_target_distribution(df, "SalePrice", "ames_housing", output_dir)


def preprocess_energy_efficiency(
    input_dir: Path | str = "data",
    output_dir: Path | str = "data_processed",
) -> None:
    """
    Preprocess Energy Efficiency dataset.

    Preprocessing steps:
    1. Exclude Relative Compactness and Wall Area (per data_descriptions/energy_efficiency.md)
    2. Create train/val/test splits (same splits for both targets)
    3. Save two separate files: one for Heating Load, one for Cooling Load

    Args:
        input_dir: Directory containing raw data
        output_dir: Directory to save processed data
    """
    input_path = Path(input_dir) / "energy_efficiency.csv"
    df = pd.read_csv(input_path)

    # Exclude variables per specification
    exclude_vars = ["Relative Compactness", "Wall Area"]
    df = df.drop(columns=[col for col in exclude_vars if col in df.columns])

    # Create splits once (same splits for both targets)
    df = create_splits(df, train_size=0.7, val_size=0.2, test_size=0.1, random_state=42)

    # Save two versions - one for each target
    # Version 1: Heating Load as target
    heating_df = df.drop(columns=["Cooling Load"])
    save_processed_data(heating_df, "energy_efficiency_heating", output_dir)
    plot_target_distribution(heating_df, "Heating Load", "energy_efficiency_heating", output_dir)

    # Version 2: Cooling Load as target
    cooling_df = df.drop(columns=["Heating Load"])
    save_processed_data(cooling_df, "energy_efficiency_cooling", output_dir)
    plot_target_distribution(cooling_df, "Cooling Load", "energy_efficiency_cooling", output_dir)


def preprocess_california_housing(
    input_dir: Path | str = "data",
    output_dir: Path | str = "data_processed",
) -> None:
    """
    Preprocess California Housing dataset.

    Preprocessing steps:
    1. Remove houses with median_house_value > 500,000 (censored values)
    2. Create train/val/test splits

    Args:
        input_dir: Directory containing raw data
        output_dir: Directory to save processed data
    """
    input_path = Path(input_dir) / "california_housing.csv"
    df = pd.read_csv(input_path)

    # Remove censored values (houses capped at $500k+)
    df = df[df["median_house_value"] <= 500000]

    # Create splits (no stratification for continuous targets)
    df = create_splits(df, train_size=0.7, val_size=0.2, test_size=0.1, random_state=42)

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
    preprocess_energy_efficiency()
