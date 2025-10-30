"""
Configuration file for Wooldridge Econometrics exercises
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Dataset URLs
DATASET_BASE_URL = "https://faculty.utrgv.edu/diego.escobari/teaching"
DATASET_INFO_URL = f"{DATASET_BASE_URL}/Datasets.html"

# Plotting settings
PLOT_CONFIG = {
    'figure_size': (10, 6),
    'style': 'seaborn-v0_8-darkgrid',
    'dpi': 100,
    'font_size': 12,
    'save_format': 'png'
}

# Statistical settings
STATS_CONFIG = {
    'significance_level': 0.05,
    'confidence_level': 0.95,
    'decimal_places': 4
}

# GPU settings
GPU_CONFIG = {
    'use_gpu': False,  # Set to True if you have GPU and packages installed
    'gpu_threshold': 100000,  # Use GPU for datasets larger than this many rows
}

# Create directories if they don't exist
for directory in [DATA_DIR, NOTEBOOKS_DIR, OUTPUTS_DIR]:
    directory.mkdir(exist_ok=True)
