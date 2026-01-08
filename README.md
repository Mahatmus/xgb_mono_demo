# XGBoost Monotonic Constraints Demonstration

Demonstrating monotonic constraint techniques in XGBoost using 6 curated regression datasets.

## Dataset Setup

### Prerequisites

1. **Install dependencies**:
   ```bash
   uv sync
   ```

   Or for simpler setup without virtual environment management:
   ```bash
   uv pip install .
   ```

2. **Configure Kaggle API**:
   - Create a Kaggle account at https://www.kaggle.com
   - Go to https://www.kaggle.com/settings/account → API → "Generate New Token"
   - Copy `.env.template` to `.env` and add your token:
     ```bash
     cp .env.template .env
     # Edit .env and replace 'your-token-here' with your actual token
     ```

3. **Accept Kaggle Competition Rules**:
   - Visit https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
   - Click "Join Competition" and accept rules (required for Ames Housing download)

### Download Datasets

```bash
python data_init.py
```

Or if using uv's managed environment:
```bash
uv run data_init.py
```

This creates a `data/` directory with 6 CSV files:
- `california_housing.csv` (20,640 rows, 8 features)
- `diamonds.csv` (53,940 rows, 10 features)
- `auto_mpg.csv` (398 rows, 7 features)
- `insurance.csv` (1,338 rows, 7 features)
- `concrete_strength.csv` (1,030 rows, 8 features)
- `ames_housing.csv` (2,930 rows, 79 features)

## Datasets Overview

See [datasets.md](datasets.md) for detailed information about each dataset, including:
- Source and documentation
- Key monotonic variables
- Domain-specific justifications for constraints
- Statistical characteristics

## Project Structure

```
xgboost_mono/
├── data_init.py       # Dataset download script
├── datasets.md        # Dataset documentation
├── pyproject.toml     # Project dependencies
├── README.md          # This file
└── data/              # Downloaded datasets (created by data_init.py)
```
