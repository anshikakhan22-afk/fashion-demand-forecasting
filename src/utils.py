"""Utility functions and path configurations for Fashion Demand Forecasting.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Return directory path for data storage."""
    data_dir = get_project_root() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_raw_data_dir() -> Path:
    """Return directory path for raw incoming data."""
    raw_dir = get_data_dir() / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def get_processed_data_dir() -> Path:
    """Return directory path for processed datasets."""
    proc_dir = get_data_dir() / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    return proc_dir


def get_sample_data_dir() -> Path:
    """Return directory path for sample fallback datasets."""
    sample_dir = get_data_dir() / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir


def get_models_dir() -> Path:
    """Return directory path for trained model artifacts."""
    models_dir = get_project_root() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_reports_dir() -> Path:
    """Return directory path for reports and analytical exports."""
    reports_dir = get_project_root() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def setup_logger(name: str = "fashion_forecast", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a standardized console/file logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def set_seed(seed: int = 42) -> None:
    """Set global random seed for numpy, random, and environment."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def format_metric(val: float | None, unit: str = "", decimals: int = 2) -> str:
    """Format floating point numbers safely with units."""
    if val is None or np.isnan(val) or np.isinf(val):
        return "N/A"
    return f"{val:.{decimals}f} {unit}".strip()
