import requests
from io import BytesIO
def to_bool_like(val):
    """
    Robustly convert Excel bool-like values to Python bool or pd.NA.
    Accepts: 1, 0, '1', '0', 'TRUE', 'FALSE', 'Yes', 'No', True, False, etc.
    Returns: True, False, or pd.NA
    """
    if pd.isna(val):
        return pd.NA
    sval = str(val).strip().lower()
    if sval in ['1', 'true', 'yes', 'y', 't']:
        return True
    if sval in ['0', 'false', 'no', 'n', 'f']:
        return False
    return pd.NA
"""
Econometrics Utility Functions for Wooldridge Exercises

This module provides helper functions for:
- Data loading and management
- Regression analysis (OLS, 2SLS, panel models)
- Statistical tests and diagnostics
- Visualization
- GPU acceleration (optional)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict
import warnings
import json
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Try to import GPU libraries
try:
    import cupy as cp
    import cudf
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Import config
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_DIR, OUTPUTS_DIR, PLOT_CONFIG, STATS_CONFIG, GPU_CONFIG,
    DATASET_BASE_URL
)

# Set default plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ============================================================================
# GPU Helper Functions
# ============================================================================

def check_gpu_available() -> bool:
    """Check if GPU is available and configured"""
    return GPU_AVAILABLE and GPU_CONFIG.get('use_gpu', False)


def to_gpu(data: Union[pd.DataFrame, np.ndarray]):
    """Convert data to GPU format if available"""
    if not check_gpu_available():
        return data
    
    if isinstance(data, pd.DataFrame):
        return cudf.from_pandas(data)
    elif isinstance(data, np.ndarray):
        return cp.array(data)
    return data


def to_cpu(data):
    """Convert GPU data back to CPU format"""
    if not GPU_AVAILABLE:
        return data
    
    if isinstance(data, cudf.DataFrame):
        return data.to_pandas()
    elif hasattr(data, 'get'):  # CuPy array
        return data.get()
    return data


# ============================================================================
# Data Loading Functions
# ============================================================================

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_column_mappings() -> Dict:
    """
    Load column mappings from JSON file.
    
    Returns:
    --------
    dict
        Dictionary containing column mappings for all datasets
    """
    mappings_file = DATA_DIR / 'column_mappings.json'
    with open(mappings_file, 'r') as f:
        return json.load(f)


def get_pandas_dtype(type_str: str):
    """
    Convert our type strings to pandas dtypes.
    
    Parameters:
    -----------
    type_str : str
        Type string from column mappings
        
    Returns:
    --------
    pandas dtype
    """
    type_mapping = {
        'integer': 'float64',    # Use float64 to handle missing values
        'float': 'float64',
        'boolean': 'boolean',    # Pandas nullable boolean
        'string': 'string'
    }
    return type_mapping.get(type_str, 'object')


def convert_excel_to_clean_csv(dataset_name: str, data_dir: Path = DATA_DIR, force_convert: bool = False) -> bool:
    """
    Convert Excel file to clean CSV with proper data types.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    data_dir : Path
        Directory containing the data files
    force_convert : bool
        Whether to reconvert even if CSV already exists
        
    Returns:
    --------
    bool
        True if conversion successful
    """

    excel_file = data_dir / f"{dataset_name}.xls"
    parquet_file = data_dir / f"{dataset_name}_compute.parquet"

    # Check if conversion needed
    if parquet_file.exists() and not force_convert:
        print(f"Compute Parquet already exists: {parquet_file}")
        return True

    # Try to load Excel from disk, else download in memory
    if excel_file.exists():
        print(f"Reading Excel from disk: {excel_file}")
        excel_source = excel_file
        read_excel = lambda **kwargs: pd.read_excel(excel_file, **kwargs)
    else:
        url = f"https://faculty.utrgv.edu/diego.escobari/teaching/Datasets/{dataset_name.upper()}.xls"
        print(f"Downloading Excel in memory from {url}")
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"Failed to download Excel for {dataset_name}")
            return False
        excel_bytes = BytesIO(resp.content)
        excel_source = excel_bytes
        read_excel = lambda **kwargs: pd.read_excel(excel_bytes, **kwargs)

    try:
        # Load column mappings
        mappings = load_column_mappings()
        if dataset_name not in mappings:
            print(f"No column mappings found for dataset: {dataset_name}")
            return False

        dataset_meta = mappings[dataset_name]

        # Step 1: Read Excel file as strings to preserve original values
        print(f"Converting {dataset_name} Excel to compute Parquet...")
        df = read_excel(header=None, dtype=str)

        # Step 2: Assign column names
        expected_cols = len(dataset_meta['columns'])
        if len(df.columns) == expected_cols:
            df.columns = list(dataset_meta['columns'].keys())
        else:
            print(f"Warning: Expected {expected_cols} columns, found {len(df.columns)}")
            # Try reading with header
            df = read_excel(header=0, dtype=str)

        # Step 3: Convert data types based on metadata
        for col, meta in dataset_meta['columns'].items():
            if col in df.columns:
                if meta['type'] in ['integer', 'float']:
                    # Convert to numeric, empty strings become NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif meta['type'] == 'boolean':
                    # Debug: print unique values before conversion
                    print(f"[DEBUG] Unique values in boolean column '{col}' before conversion:", df[col].unique())
                    # Always use robust bool-like conversion, even for 0/1
                    df[col] = df[col].apply(to_bool_like).astype('boolean')

        # Step 4: Save as compute Parquet
        df.to_parquet(parquet_file, index=False)
        print(f"Successfully converted to: {parquet_file}")
        return True

    except Exception as e:
        print(f"Error converting {excel_file}: {e}")
        return False
def load_compute_parquet(dataset_name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Load compute-ready Parquet file for a dataset.
    """
    parquet_file = data_dir / f"{dataset_name}_compute.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Compute Parquet not found: {parquet_file}")
    return pd.read_parquet(parquet_file)


def apply_dataset_cleaning_rules(df: pd.DataFrame, dataset_name: str, missing_value_rules: dict) -> pd.DataFrame:
    """
    Apply dataset-specific cleaning rules to handle missing values and data quality issues.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The loaded dataframe
    dataset_name : str
        Name of the dataset
    missing_value_rules : dict
        Dictionary containing cleaning rules for each dataset
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    if dataset_name.lower() not in missing_value_rules:
        return df
    
    df_cleaned = df.copy()
    rules = missing_value_rules[dataset_name.lower()]
    
    for column, column_rules in rules.items():
        if column in df_cleaned.columns:
            if column_rules.get('replace_zeros_with_nan', False):
                # Replace zeros with NaN for this column
                zeros_count = (df_cleaned[column] == 0).sum()
                if zeros_count > 0:
                    print(f"Replacing {zeros_count} zero values with NaN in column '{column}'")
                    df_cleaned[column] = df_cleaned[column].replace(0, np.nan)
    
    return df_cleaned


def load_wooldridge_data(
    dataset_name: str,
    data_dir: Path = DATA_DIR,
    use_gpu: bool = False,
    auto_download: bool = True,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Load a Wooldridge dataset using CSV files with metadata.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (without extension), e.g., 'wage1', 'hprice1'
    data_dir : Path
        Directory where datasets are stored
    use_gpu : bool
        Whether to return GPU-accelerated DataFrame (cuDF)
    auto_download : bool
        Whether to automatically download if file not found
    force_download : bool
        Whether to force redownload of the dataset
    
    Returns:
    --------
    pd.DataFrame or cudf.DataFrame
        Loaded dataset
    
    Example:
    --------
    >>> df = load_wooldridge_data('wage1')
    >>> print(df.head())
    """
    
    parquet_file = data_dir / f"{dataset_name}_compute.parquet"

    # Step 1: Check if Parquet exists, if not try to download and convert
    if not parquet_file.exists() or force_download:
        if auto_download:
            print(f"Compute Parquet file not found for {dataset_name}. Downloading and converting...")
            if not download_wooldridge_data(dataset_name, data_dir, force_download):
                raise FileNotFoundError(f"Could not download {dataset_name}")
            # After download, convert Excel to Parquet
            if not convert_excel_to_clean_csv(dataset_name, data_dir, force_convert=True):
                raise FileNotFoundError(f"Could not convert {dataset_name} to Parquet")
        else:
            raise FileNotFoundError(f"Compute Parquet file not found: {parquet_file}")

    # Step 2: Load Parquet
    try:
        df = pd.read_parquet(parquet_file)
        print(f"Loaded {dataset_name} from {parquet_file}")
        print(f"Shape: {df.shape}")
        if use_gpu and check_gpu_available():
            return to_gpu(df)
        return df
    except Exception as e:
        print(f"Error loading Parquet {parquet_file}: {e}")
        raise


def load_wooldridge_data_legacy(
    dataset_name: str,
    data_dir: Path = DATA_DIR,
    use_gpu: bool = False,
    auto_download: bool = True
) -> pd.DataFrame:
    """
    Legacy Excel-based data loading (kept for fallback).
    """
    
    # Define column names for datasets that don't have headers
    column_mappings = {
        'wage1': ['wage', 'educ', 'exper', 'tenure', 'nonwhite', 'female', 'married', 
                  'numdep', 'smsa', 'northcen', 'south', 'west', 'construc', 'ndurman', 
                  'trcommpu', 'trade', 'services', 'profserv', 'profocc', 'clerocc', 
                  'servocc', 'lwage', 'expersq', 'tenursq'],
        'bwght': ['faminc', 'cigtax', 'cigprice', 'bwght', 'fatheduc', 'motheduc', 
                  'parity', 'male', 'white', 'cigs', 'lbwght', 'bwghtlbs', 'packs', 
                  'lfaminc'],
        'meap01': ['dcode', 'bcode', 'math4', 'read4', 'lunch', 'enroll', 'expend',
                   'exppp', 'lenroll', 'lexpend', 'lexppp']
    }
    
    # Define dataset-specific cleaning rules for missing values
    missing_value_rules = {
        'bwght': {
            'fatheduc': {'replace_zeros_with_nan': True},  # Zero years of education should be NaN
            'motheduc': {'replace_zeros_with_nan': True}   # Zero years of education should be NaN
        }
    }
    
    # Try different extensions
    extensions = ['.xls', '.xlsx', '.dta', '.csv']
    
    for ext in extensions:
        file_path = data_dir / f"{dataset_name}{ext}"
        if file_path.exists():
            if ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path, header=None)  # Read without headers first
                # Apply column names if we have them
                if dataset_name.lower() in column_mappings:
                    expected_cols = len(column_mappings[dataset_name.lower()])
                    if len(df.columns) == expected_cols:
                        df.columns = column_mappings[dataset_name.lower()]
                    else:
                        print(f"Warning: Expected {expected_cols} columns but found {len(df.columns)}")
                        df = pd.read_excel(file_path, header=0)  # Fallback to using first row as headers
                else:
                    df = pd.read_excel(file_path, header=0)  # Use first row as headers for unknown datasets
            elif ext == '.dta':
                df = pd.read_stata(file_path)
            elif ext == '.csv':
                df = pd.read_csv(file_path)
            
            print(f"Loaded {dataset_name} from {file_path}")
            print(f"Shape: {df.shape}")
            
            # Apply dataset-specific cleaning rules
            df = apply_dataset_cleaning_rules(df, dataset_name, missing_value_rules)
            
            if use_gpu and check_gpu_available():
                return to_gpu(df)
            return df
    
    # If file not found, try to download if auto_download is True
    if auto_download:
        print(f"Dataset '{dataset_name}' not found. Attempting to download...")
        
        import urllib.request
        from urllib.error import URLError
        
        # Ensure data directory exists
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the file
        download_url = f"{DATASET_BASE_URL}/{dataset_name.upper()}.xls"
        file_path = data_dir / f"{dataset_name}.xls"
        
        try:
            print(f"Downloading {dataset_name} from {download_url}...")
            urllib.request.urlretrieve(download_url, file_path)
            print(f"Successfully downloaded to {file_path}")
            
            # Now load the downloaded file
            df = pd.read_excel(file_path, header=None)  # Read without headers first
            # Apply column names if we have them
            if dataset_name.lower() in column_mappings:
                expected_cols = len(column_mappings[dataset_name.lower()])
                if len(df.columns) == expected_cols:
                    df.columns = column_mappings[dataset_name.lower()]
                else:
                    print(f"Warning: Expected {expected_cols} columns but found {len(df.columns)}")
                    df = pd.read_excel(file_path, header=0)  # Fallback
            else:
                df = pd.read_excel(file_path, header=0)  # Use first row as headers for unknown datasets
                
            print(f"Loaded {dataset_name} from {file_path}")
            print(f"Shape: {df.shape}")
            
            # Apply dataset-specific cleaning rules
            df = apply_dataset_cleaning_rules(df, dataset_name, missing_value_rules)
            
            if use_gpu and check_gpu_available():
                return to_gpu(df)
            return df
            
        except URLError as e:
            print(f"Failed to download {dataset_name}: {e}")
            download_url = f"{DATASET_BASE_URL}/{dataset_name.upper()}.xls"
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' not found in {data_dir.resolve()}\n"
                f"Auto-download failed. Please manually download from: {download_url}\n"
                f"And save it to: {data_dir.resolve()}"
            )
    else:
        # If auto_download is False, just raise the error
        download_url = f"{DATASET_BASE_URL}/{dataset_name.upper()}.xls"
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found in {data_dir.resolve()}\n"
            f"Please download from: {download_url}\n"
            f"And save it to: {data_dir.resolve()}"
        )


def download_wooldridge_data(
    dataset_name: str,
    data_dir: Path = DATA_DIR,
    force_download: bool = False
) -> bool:
    """
    Download a Wooldridge dataset from the web and save as CSV.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (without extension), e.g., 'wage1', 'hprice1'
    data_dir : Path
        Directory where datasets should be saved
    force_download : bool
        Whether to download even if file already exists
    
    Returns:
    --------
    bool
        True if download was successful, False otherwise
    
    Example:
    --------
    >>> download_wooldridge_data('wage1')
    >>> df = load_wooldridge_data('wage1')
    """
    import urllib.request
    from urllib.error import URLError
    import tempfile
    
    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if clean CSV already exists
    csv_file = data_dir / f"{dataset_name}.csv"
    if csv_file.exists() and not force_download:
        print(f"Dataset '{dataset_name}' already exists at {csv_file}")
        return True
    
    # Download to temporary Excel file first
    download_url = f"{DATASET_BASE_URL}/{dataset_name.upper()}.xls"
    
    try:
        print(f"Downloading {dataset_name} from {download_url}...")
        
        # Create temporary file for Excel data
        with tempfile.NamedTemporaryFile(suffix='.xls', delete=False) as temp_file:
            urllib.request.urlretrieve(download_url, temp_file.name)
            temp_excel_path = temp_file.name
        
        # Load column mappings
        mappings = load_column_mappings()
        if dataset_name not in mappings:
            print(f"Warning: No column mappings found for {dataset_name}")
            # Save as basic CSV without type conversion
            df = pd.read_excel(temp_excel_path, header=None)
            df.to_csv(csv_file, index=False)
        else:
            # Convert with proper types
            dataset_meta = mappings[dataset_name]
            df = pd.read_excel(temp_excel_path, header=None, dtype=str)
            
            # Assign column names
            expected_cols = len(dataset_meta['columns'])
            if len(df.columns) == expected_cols:
                df.columns = list(dataset_meta['columns'].keys())
            else:
                print(f"Warning: Expected {expected_cols} columns, found {len(df.columns)}")
            
            # Convert data types
            for col, meta in dataset_meta['columns'].items():
                if col in df.columns:
                    if meta['type'] in ['integer', 'float']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if meta['type'] == 'integer':
                            df[col] = df[col].astype('Int64')
                    elif meta['type'] == 'boolean':
                        # Convert 0/1 to boolean, handling NaN properly
                        df[col] = df[col].astype('boolean')
            
            # Save as CSV
            df.to_csv(csv_file, index=False)
        
        # Clean up temporary file
        Path(temp_excel_path).unlink()
        
        print(f"Successfully downloaded and converted to: {csv_file}")
        return True
        
    except URLError as e:
        print(f"Failed to download {dataset_name}: {e}")
        print(f"Please manually download from: {download_url}")
        return False
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")
        return False


def load_wooldridge_data_with_download(
    dataset_name: str,
    data_dir: Path = DATA_DIR,
    use_gpu: bool = False,
    auto_download: bool = True
) -> pd.DataFrame:
    """
    Load a Wooldridge dataset, downloading it first if needed.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (without extension)
    data_dir : Path
        Directory where datasets are stored
    use_gpu : bool
        Whether to return GPU-accelerated DataFrame
    auto_download : bool
        Whether to automatically download if file not found
    
    Returns:
    --------
    pd.DataFrame or cudf.DataFrame
        Loaded dataset
    """
    try:
        return load_wooldridge_data(dataset_name, data_dir, use_gpu)
    except FileNotFoundError:
        if auto_download:
            print(f"Dataset '{dataset_name}' not found. Attempting to download...")
            if download_wooldridge_data(dataset_name, data_dir):
                return load_wooldridge_data(dataset_name, data_dir, use_gpu)
            else:
                raise
        else:
            raise


def get_dataset_info(dataset_name: str) -> str:
    """
    Get information about variables in a dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    
    Returns:
    --------
    str
        Variable descriptions if available
    """
    info_file = DATA_DIR / f"{dataset_name}.txt"
    
    if info_file.exists():
        with open(info_file, 'r') as f:
            return f.read()
    else:
        info_url = f"{DATASET_BASE_URL}/{dataset_name}.txt"
        return f"Info file not found. Try downloading from: {info_url}"


# ============================================================================
# Regression Functions
# ============================================================================

def run_ols(
    data: pd.DataFrame,
    dependent: str,
    independent: List[str],
    add_constant: bool = True,
    robust: bool = False
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run OLS regression with clean output.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing variables
    dependent : str
        Name of dependent variable
    independent : List[str]
        List of independent variable names
    add_constant : bool
        Whether to add intercept (default: True)
    robust : bool
        Whether to use heteroskedasticity-robust standard errors (default: False)
    
    Returns:
    --------
    RegressionResultsWrapper
        Regression results object
    
    Example:
    --------
    >>> results = run_ols(df, 'wage', ['educ', 'exper', 'tenure'])
    >>> print(results.summary())
    """
    # Prepare data
    y = data[dependent]
    X = data[independent]
    
    if add_constant:
        X = sm.add_constant(X)
    
    # Fit model
    if robust:
        model = sm.OLS(y, X)
        results = model.fit(cov_type='HC1')  # White's robust standard errors
    else:
        model = sm.OLS(y, X)
        results = model.fit()
    
    return results


def run_ols_formula(
    formula: str,
    data: pd.DataFrame,
    robust: bool = False
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run OLS using formula notation (R-style).
    
    Parameters:
    -----------
    formula : str
        Regression formula, e.g., 'wage ~ educ + exper + tenure'
    data : pd.DataFrame
        Dataset
    robust : bool
        Use robust standard errors
    
    Returns:
    --------
    RegressionResultsWrapper
        Regression results
    
    Example:
    --------
    >>> results = run_ols_formula('wage ~ educ + exper + I(exper**2)', df)
    """
    if robust:
        results = smf.ols(formula, data=data).fit(cov_type='HC1')
    else:
        results = smf.ols(formula, data=data).fit()
    
    return results


def pretty_print_results(
    results: sm.regression.linear_model.RegressionResultsWrapper,
    decimal_places: int = 4
) -> None:
    """
    Print regression results in a clean, readable format.
    
    Parameters:
    -----------
    results : RegressionResultsWrapper
        Fitted regression results
    decimal_places : int
        Number of decimal places to display
    """
    print("=" * 80)
    print("REGRESSION RESULTS")
    print("=" * 80)
    print(f"\nDependent Variable: {results.model.endog_names}")
    print(f"Number of Observations: {int(results.nobs)}")
    print(f"R-squared: {results.rsquared:.{decimal_places}f}")
    print(f"Adjusted R-squared: {results.rsquared_adj:.{decimal_places}f}")
    print(f"F-statistic: {results.fvalue:.{decimal_places}f}")
    print(f"Prob (F-statistic): {results.f_pvalue:.{decimal_places}e}")
    print("\n" + "-" * 80)
    print(f"{'Variable':<15} {'Coef':<12} {'Std Err':<12} {'t':<10} {'P>|t|':<10}")
    print("-" * 80)
    
    for var in results.params.index:
        coef = results.params[var]
        se = results.bse[var]
        t_stat = results.tvalues[var]
        p_val = results.pvalues[var]
        
        sig = ''
        if p_val < 0.01:
            sig = '***'
        elif p_val < 0.05:
            sig = '**'
        elif p_val < 0.1:
            sig = '*'
        
        print(f"{var:<15} {coef:<12.{decimal_places}f} {se:<12.{decimal_places}f} "
            f"{t_stat:<10.{decimal_places}f} {p_val:<10.{decimal_places}f} {sig}")
    
    print("-" * 80)
    print("Significance: *** p<0.01, ** p<0.05, * p<0.1")
    print("=" * 80)


# ============================================================================
# Diagnostic Tests
# ============================================================================

def check_heteroskedasticity(
    results: sm.regression.linear_model.RegressionResultsWrapper
) -> Dict[str, Tuple[float, float]]:
    """
    Perform heteroskedasticity tests (Breusch-Pagan and White).
    
    Parameters:
    -----------
    results : RegressionResultsWrapper
        Fitted regression results
    
    Returns:
    --------
    dict
        Test statistics and p-values
    """
    # Breusch-Pagan test
    bp_test = het_breuschpagan(results.resid, results.model.exog)
    
    # White test
    white_test = het_white(results.resid, results.model.exog)
    
    print("\n" + "=" * 80)
    print("HETEROSKEDASTICITY TESTS")
    print("=" * 80)
    print(f"\nBreusch-Pagan Test:")
    print(f"  LM Statistic: {bp_test[0]:.4f}")
    print(f"  P-value: {bp_test[1]:.4f}")
    print(f"  Result: {'Heteroskedasticity detected' if bp_test[1] < 0.05 else 'No heteroskedasticity'}")
    
    print(f"\nWhite Test:")
    print(f"  LM Statistic: {white_test[0]:.4f}")
    print(f"  P-value: {white_test[1]:.4f}")
    print(f"  Result: {'Heteroskedasticity detected' if white_test[1] < 0.05 else 'No heteroskedasticity'}")
    print("=" * 80)
    
    return {
        'breusch_pagan': (bp_test[0], bp_test[1]),
        'white': (white_test[0], white_test[1])
    }


def check_multicollinearity(
    data: pd.DataFrame,
    independent: List[str]
) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factors (VIF) for multicollinearity detection.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    independent : List[str]
        List of independent variables
    
    Returns:
    --------
    pd.DataFrame
        VIF values for each variable
    """
    X = data[independent]
    X = sm.add_constant(X)
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    print("\n" + "=" * 80)
    print("VARIANCE INFLATION FACTORS (VIF)")
    print("=" * 80)
    print("\nRule of thumb: VIF > 10 indicates high multicollinearity")
    print(vif_data.to_string(index=False))
    print("=" * 80)
    
    return vif_data


def durbin_watson_test(
    results: sm.regression.linear_model.RegressionResultsWrapper
) -> float:
    """
    Perform Durbin-Watson test for autocorrelation.
    
    Parameters:
    -----------
    results : RegressionResultsWrapper
        Fitted regression results
    
    Returns:
    --------
    float
        Durbin-Watson statistic
    """
    dw_stat = durbin_watson(results.resid)
    
    print("\n" + "=" * 80)
    print("DURBIN-WATSON TEST FOR AUTOCORRELATION")
    print("=" * 80)
    print(f"\nDurbin-Watson Statistic: {dw_stat:.4f}")
    print("\nInterpretation:")
    print("  DW ≈ 2: No autocorrelation")
    print("  DW < 2: Positive autocorrelation")
    print("  DW > 2: Negative autocorrelation")
    print("  Rule of thumb: 1.5 < DW < 2.5 is acceptable")
    print("=" * 80)
    
    return dw_stat


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_residuals(
    results: sm.regression.linear_model.RegressionResultsWrapper,
    save_path: Optional[Path] = None
) -> None:
    """
    Create diagnostic plots for regression residuals.
    
    Parameters:
    -----------
    results : RegressionResultsWrapper
        Fitted regression results
    save_path : Path, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs Fitted
    axes[0, 0].scatter(results.fittedvalues, results.resid, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    # Q-Q plot
    stats.probplot(results.resid, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot')
    
    # Scale-Location plot
    standardized_resid = np.sqrt(np.abs(results.resid_pearson))
    axes[1, 0].scatter(results.fittedvalues, standardized_resid, alpha=0.5)
    axes[1, 0].set_xlabel('Fitted values')
    axes[1, 0].set_ylabel('√|Standardized residuals|')
    axes[1, 0].set_title('Scale-Location Plot')
    
    # Residuals histogram
    axes[1, 1].hist(results.resid, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Histogram of Residuals')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_correlation_matrix(
    data: pd.DataFrame,
    variables: Optional[List[str]] = None,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot correlation matrix heatmap.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    variables : List[str], optional
        Specific variables to include (default: all numeric)
    save_path : Path, optional
        Path to save the figure
    """
    if variables:
        corr = data[variables].corr()
    else:
        corr = data.select_dtypes(include=[np.number]).corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def summary_statistics(
    data: pd.DataFrame,
    variables: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate comprehensive summary statistics.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    variables : List[str], optional
        Specific variables to summarize
    
    Returns:
    --------
    pd.DataFrame
        Summary statistics
    """
    if variables:
        summary = data[variables].describe()
    else:
        summary = data.describe()
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(summary.to_string())
    print("=" * 80)
    
    return summary


# ============================================================================
# Hypothesis Testing Helpers
# ============================================================================

def test_linear_restriction(
    results: sm.regression.linear_model.RegressionResultsWrapper,
    restriction: str
) -> None:
    """
    Test linear restrictions on coefficients (F-test).
    
    Parameters:
    -----------
    results : RegressionResultsWrapper
        Fitted regression results
    restriction : str
        Restriction to test, e.g., 'educ = exper' or 'educ = 0, exper = 0'
    
    Example:
    --------
    >>> test_linear_restriction(results, 'educ = exper')
    """
    f_test = results.f_test(restriction)
    
    print("\n" + "=" * 80)
    print("LINEAR RESTRICTION TEST")
    print("=" * 80)
    print(f"\nRestriction: {restriction}")
    print(f"F-statistic: {f_test.fvalue[0][0]:.4f}")
    print(f"P-value: {f_test.pvalue:.4f}")
    print(f"Result: {'Reject H0' if f_test.pvalue < 0.05 else 'Fail to reject H0'}")
    print("=" * 80)


# ============================================================================
# GPU-Accelerated Functions (for intensive computations)
# ============================================================================

def bootstrap_ols(
    data: pd.DataFrame,
    dependent: str,
    independent: List[str],
    n_iterations: int = 1000,
    use_gpu: bool = False
) -> Dict[str, np.ndarray]:
    """
    Bootstrap OLS regression coefficients.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    dependent : str
        Dependent variable name
    independent : List[str]
        Independent variable names
    n_iterations : int
        Number of bootstrap iterations
    use_gpu : bool
        Use GPU acceleration if available
    
    Returns:
    --------
    dict
        Bootstrap coefficient distributions
    """
    y = data[dependent].values
    X = sm.add_constant(data[independent]).values
    n = len(y)
    
    # Storage for coefficients
    coefs = []
    
    if use_gpu and check_gpu_available():
        print(f"Running bootstrap on GPU with {n_iterations} iterations...")
        y_gpu = cp.array(y)
        X_gpu = cp.array(X)
        
        for i in range(n_iterations):
            # Bootstrap sample
            idx = cp.random.choice(n, size=n, replace=True)
            y_boot = y_gpu[idx]
            X_boot = X_gpu[idx]
            
            # OLS: β = (X'X)^(-1) X'y
            beta = cp.linalg.inv(X_boot.T @ X_boot) @ X_boot.T @ y_boot
            coefs.append(beta.get())
    else:
        print(f"Running bootstrap on CPU with {n_iterations} iterations...")
        for i in range(n_iterations):
            # Bootstrap sample
            idx = np.random.choice(n, size=n, replace=True)
            y_boot = y[idx]
            X_boot = X[idx]
            
            # OLS
            beta = np.linalg.inv(X_boot.T @ X_boot) @ X_boot.T @ y_boot
            coefs.append(beta)
    
    coefs = np.array(coefs)
    var_names = ['const'] + independent
    
    results = {var: coefs[:, i] for i, var in enumerate(var_names)}
    
    print(f"Bootstrap completed. {n_iterations} iterations.")
    return results


def plot_bootstrap_distribution(
    bootstrap_results: Dict[str, np.ndarray],
    variable: str
) -> None:
    """
    Plot bootstrap distribution for a specific variable.
    
    Parameters:
    -----------
    bootstrap_results : dict
        Results from bootstrap_ols()
    variable : str
        Variable name to plot
    """
    coefs = bootstrap_results[variable]
    
    plt.figure(figsize=(10, 6))
    plt.hist(coefs, bins=50, density=True, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(coefs), color='r', linestyle='--', label=f'Mean: {np.mean(coefs):.4f}')
    plt.axvline(np.percentile(coefs, 2.5), color='g', linestyle='--', label='95% CI')
    plt.axvline(np.percentile(coefs, 97.5), color='g', linestyle='--')
    plt.xlabel(f'Coefficient for {variable}')
    plt.ylabel('Density')
    plt.title(f'Bootstrap Distribution: {variable}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"\n{variable} Bootstrap Statistics:")
    print(f"  Mean: {np.mean(coefs):.4f}")
    print(f"  Std Dev: {np.std(coefs):.4f}")
    print(f"  95% CI: [{np.percentile(coefs, 2.5):.4f}, {np.percentile(coefs, 97.5):.4f}]")


if __name__ == "__main__":
    print("Econometrics Utils Module")
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"GPU Configured: {GPU_CONFIG.get('use_gpu', False)}")
