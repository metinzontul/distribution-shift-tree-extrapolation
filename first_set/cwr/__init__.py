"""First set of experiments: leakage-free harness (data loading, shift construction, models, metrics)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"

__all__ = ["PROJECT_ROOT", "DATASETS_DIR", "RESULTS_DIR"]
