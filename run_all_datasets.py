"""
Meta script to run tune_simple_grid.py on all datasets from a config file.
"""

import argparse
import csv
import logging
import time
from pathlib import Path

import pandas as pd

from tune_simple_grid import parse_monotonic_constraints, run_grid_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> list[dict]:
    """Load dataset config from CSV file."""
    datasets = []
    with open(config_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            datasets.append({
                "dataset": row["dataset"],
                "target": row["target"],
                "monotonic": row.get("monotonic", ""),
                "objective": row.get("objective", "count:poisson"),
            })
    return datasets


def run_all(
    config_path: str,
    output_dir: str = "results",
    max_depths: list[int] | None = None,
    learning_rate: float = 0.1,
):
    """Run grid search on all datasets in config."""
    if max_depths is None:
        max_depths = list(range(1, 11))

    datasets = load_config(config_path)
    logger.info("Loaded %d datasets from %s", len(datasets), config_path)

    total_start = time.time()
    results_summary = []

    for i, ds in enumerate(datasets, 1):
        dataset_name = Path(ds["dataset"]).stem
        logger.info("=" * 70)
        logger.info("[%d/%d] Processing: %s", i, len(datasets), dataset_name)
        logger.info("=" * 70)

        # Parse monotonic constraints (semicolon-separated in CSV)
        mono_args = [m.strip() for m in ds["monotonic"].split(";") if m.strip()]
        mono_constraints = parse_monotonic_constraints(mono_args)

        try:
            start = time.time()
            run_grid_search(
                dataset_path=ds["dataset"],
                target_col=ds["target"],
                monotonic_constraints=mono_constraints,
                max_depths=max_depths,
                learning_rate=learning_rate,
                objective=ds["objective"],
                output_dir=output_dir,
            )
            elapsed = time.time() - start
            results_summary.append({
                "dataset": dataset_name,
                "status": "success",
                "time": f"{elapsed:.1f}s",
            })
            logger.info("Completed %s in %.1fs", dataset_name, elapsed)

        except Exception as e:
            logger.error("Failed %s: %s", dataset_name, str(e))
            results_summary.append({
                "dataset": dataset_name,
                "status": "failed",
                "error": str(e),
            })

    total_elapsed = time.time() - total_start
    logger.info("=" * 70)
    logger.info("ALL DATASETS COMPLETE (total time: %.1fs)", total_elapsed)
    logger.info("=" * 70)

    for r in results_summary:
        status = r["status"]
        if status == "success":
            logger.info("  %s: %s (%s)", r["dataset"], status, r["time"])
        else:
            logger.info("  %s: %s - %s", r["dataset"], status, r.get("error", ""))

    # Collect all chosen models into overall summary
    output_path = Path(output_dir)
    all_chosen = []
    for ds in datasets:
        dataset_name = Path(ds["dataset"]).stem
        grid_file = output_path / dataset_name / "grid_results.csv"
        if grid_file.exists():
            df = pd.read_csv(grid_file)
            chosen = df[df["chosen"] == True]
            all_chosen.append(chosen)

    if all_chosen:
        summary_df = pd.concat(all_chosen, ignore_index=True)
        summary_file = output_path / "overall_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.info("Saved overall summary: %s", summary_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grid search on all datasets")
    parser.add_argument("--config", type=str, default="datasets_config.csv",
                        help="Path to config CSV file")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--depths", type=int, nargs="*", default=list(range(1, 11)),
                        help="Max depths to try (default: 1-10)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")

    args = parser.parse_args()

    run_all(
        config_path=args.config,
        output_dir=args.output_dir,
        max_depths=args.depths,
        learning_rate=args.lr,
    )
