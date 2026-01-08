# Progressive Training for XGBoost with Monotonic Constraints

## Quick Start

### Installation

```bash
# Install dependencies
uv sync

# Install package in editable mode
uv pip install -e .
```

### Setup Kaggle API

1. Get your Kaggle API token:
   - Go to https://www.kaggle.com/settings/account
   - Click "Create New Token" under API section
   - Copy the generated token (starts with `KGAT_`)

2. Configure credentials:
   ```bash
   cp .env.template .env
   # Edit .env and add your token: KAGGLE_API_TOKEN=your-token-here
   ```

3. Accept competition rules:
   - Visit https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
   - Click "Join Competition" and accept the rules

### Running the Pipeline

The pipeline consists of 3 numbered scripts that should be run in order:

#### Step 1: Download Datasets

```bash
uv run python scripts/01_download_data.py
```

Downloads 5 regression datasets:
- California Housing (Kaggle)
- Auto MPG (UCI)
- Medical Insurance (Kaggle)
- Concrete Strength (UCI)
- Ames Housing (Kaggle Competition)

Output: `data/*.csv`

#### Step 2: Preprocess Data

```bash
uv run python scripts/02_preprocess.py
```

Applies dataset-specific preprocessing:
- Feature engineering (ratios, ordinal encoding)
- Creates 5-fold cross-validation splits
- Designates fold 6 as test set

Output: `data_processed/*.csv` (with `fold` column)

#### Step 3: Run Experiments

```bash
uv run python scripts/03_run_experiments.py
```

Runs grid search comparing three approaches:
1. **Baseline** - Standard XGBoost (no constraints)
2. **Monotonic** - XGBoost with monotonic constraints
3. **Progressive** - Two-phase training (our technique)

Output:
- `results/*/grid_results.csv` - All hyperparameter combinations
- `results/*/diagnostics/*.png` - Feature importance, PDP plots
- `results/overall_summary.csv` - Best models from each dataset

### Expected Runtime

- Download: ~30 seconds
- Preprocessing: ~5 seconds
- Experiments: ~15-30 minutes (all 5 datasets)

### Customizing Experiments

Run individual datasets:
```python
from src.grid_search import run_grid_search
from src.utils import parse_monotonic_constraints

run_grid_search(
    dataset_path='data_processed/auto_mpg.csv',
    target_col='mpg',
    monotonic_constraints=parse_monotonic_constraints(['weight:-1']),
    max_depths=[3, 5, 7],
    learning_rate=0.1,
    objective='count:poisson'
)
```

Or modify `config/datasets.csv` to change hyperparameters.

## Project Structure

```
.
├── scripts/              # Entry points (run these)
│   ├── 01_download_data.py
│   ├── 02_preprocess.py
│   └── 03_run_experiments.py
├── src/                  # Library code
│   ├── preprocessing/    # Data preprocessing
│   ├── training/         # Training functions
│   ├── grid_search.py    # Experiment orchestration
│   ├── visualization.py  # Plotting and diagnostics
│   └── data_loader.py    # Data utilities
├── config/
│   └── datasets.csv      # Dataset configurations
└── docs/
    └── datasets/         # Detailed dataset descriptions
```

## Datasets

See [docs/datasets.md](docs/datasets.md) for detailed information about each dataset, including source, features, and monotonic constraint justifications.

## License

MIT License - see LICENSE.md
