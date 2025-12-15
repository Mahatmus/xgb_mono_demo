# Ames Housing Dataset

**Source**: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
**Instances**: 2,930 (training: 1,460) | **Features**: 79 | **Target**: SalePrice

## Variables

### Monotonic Constraints

| Variable | Type | Description | Monotonic Constraint |
|----------|------|-------------|---------------------|
| **GrLivArea** | Continuous | Above grade living area (sq ft) | +1 |
| **LotArea** | Continuous | Lot/property size (sq ft) | +1 |

### Recode as Ratios (relative to GrLivArea, as percentages)

| Original Variable | Recoded Variable | Description |
|-------------------|------------------|-------------|
| **1stFlrSF** | **1stFlrSF_ratio** | Proportion of living area on first floor (%) |
| **TotalBsmtSF** | **TotalBsmtSF_ratio** | Basement size relative to living area (%) |
| **GarageArea** | **GarageArea_ratio** | Garage size relative to living area (%) |
| **WoodDeckSF** | **WoodDeckSF_ratio** | Wood deck size relative to living area (%) |
| **OpenPorchSF** | **OpenPorchSF_ratio** | Open porch size relative to living area (%) |

> **Note**: 2ndFlrSF excluded because 1stFlrSF + 2ndFlrSF = GrLivArea (perfectly correlated). All original variables are absolute measurements (sq ft) in the raw data, converted to percentages of GrLivArea.

### Keep (Unconstrained)

| Variable | Type | Description |
|----------|------|-------------|
| **YearBuilt** | Integer | Original construction year |
| **FullBath** | Integer | Full bathrooms above grade |
| **Fireplaces** | Integer | Number of fireplaces |
| **OverallQual** | Ordinal | Overall quality (1-10) |
| **OverallCond** | Ordinal | Overall condition (1-10) |
| **ExterQual** | Ordinal | Exterior quality |
| **KitchenQual** | Ordinal | Kitchen quality |
| **BsmtQual** | Ordinal | Basement quality |

### Ordinal Encoding

| Variable | Encoding Order (low → high) |
|----------|----------------------------|
| **OverallQual** | 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 |
| **OverallCond** | 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 |
| **ExterQual** | Po (1) → Fa (2) → TA (3) → Gd (4) → Ex (5) |
| **KitchenQual** | Po (1) → Fa (2) → TA (3) → Gd (4) → Ex (5) |
| **BsmtQual** | NA (0) → Po (1) → Fa (2) → TA (3) → Gd (4) → Ex (5) |

> **Note**: BsmtQual includes NA for homes without basements—encoded as 0.

### Exclude

| Variable | Reason |
|----------|--------|
| **2ndFlrSF** | Perfectly correlated with 1stFlrSF (they sum to GrLivArea) |
| **TotRmsAbvGrd** | Proxy for size even as ratio |
| **GarageCars** | Redundant with GarageArea |
| **YearRemodAdd** | Correlated with YearBuilt |
| **HeatingQC** | Low signal |
| **GarageQual** | Correlated with OverallQual |
| **FireplaceQu** | Redundant with Fireplaces count |

## Target Variable

| Variable | Type | Description |
|----------|------|-------------|
| **SalePrice** | Continuous | **[TARGET]** Property sale price ($) |

## Notes

**GrLivArea** and **LotArea** are the only features with enforced monotonic constraints. All else equal, more living space and more land should increase property value—this is fundamental to real estate pricing.

Size-related variables are recoded as ratios to remove collinearity with GrLivArea while preserving information about building proportions and layout.

**Full data dictionary**: https://jse.amstat.org/v19n3/decock/DataDocumentation.txt