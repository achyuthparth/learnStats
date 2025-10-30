# Wooldridge Econometrics - Computer Exercises

This repository contains solutions to computer exercises from **"Introductory Econometrics: A Modern Approach"** by Jeffrey M. Wooldridge (5th Edition).

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. GPU Support (Optional)

If you have a CUDA-capable GPU and want to use GPU acceleration:

1. Uncomment the GPU packages in `requirements.txt`
2. Install appropriate CUDA version
3. Set `use_gpu = True` in `config.py`

**Note**: For most Wooldridge exercises (small-medium datasets), CPU is sufficient. GPU is beneficial for:
- Large simulations
- Bootstrap with many iterations
- Custom intensive computations

## Directory Structure

```
Practice/
├── data/              # Downloaded datasets (.xls files)
├── notebooks/         # Jupyter notebooks for each chapter
├── utils/             # Utility functions and helpers
├── outputs/           # Generated figures and results
├── config.py          # Configuration settings
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Usage

### Loading Datasets

```python
from utils.econometrics_utils import load_wooldridge_data

# Load a dataset
df = load_wooldridge_data('wage1')
```

### Running Regressions

```python
from utils.econometrics_utils import run_ols

# Simple OLS regression
results = run_ols(df, dependent='wage', independent=['educ', 'exper', 'tenure'])
```

## Notebooks

Each chapter has its own notebook in the `notebooks/` directory:
- `chapter_01.ipynb` - Chapter 1 exercises
- `chapter_02.ipynb` - Chapter 2 exercises
- etc.

## Data Sources

Datasets are from: https://faculty.utrgv.edu/diego.escobari/teaching/Datasets.html

Each dataset includes:
- `.txt` file: Variable descriptions
- `.xls` file: Actual data

## Resources

- [Wooldridge Textbook Website](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge/9781337558860/)
- [Statsmodels Documentation](https://www.statsmodels.org/)
- [RAPIDS cuDF Documentation](https://docs.rapids.ai/api/cudf/stable/) (for GPU)
