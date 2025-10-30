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

def load_wooldridge_data(
    dataset_name: str,
    data_dir: Path = DATA_DIR,
    use_gpu: bool = False
) -> pd.DataFrame:
    """
    Load a Wooldridge dataset from local storage or download if needed.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (without extension), e.g., 'wage1', 'hprice1'
    data_dir : Path
        Directory where datasets are stored
    use_gpu : bool
        Whether to return GPU-accelerated DataFrame (cuDF)
    
    Returns:
    --------
    pd.DataFrame or cudf.DataFrame
        Loaded dataset
    
    Example:
    --------
    >>> df = load_wooldridge_data('wage1')
    >>> print(df.head())
    """
    # Try different extensions
    extensions = ['.xls', '.xlsx', '.dta', '.csv']
    
    for ext in extensions:
        file_path = data_dir / f"{dataset_name}{ext}"
        if file_path.exists():
            if ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path)
            elif ext == '.dta':
                df = pd.read_stata(file_path)
            elif ext == '.csv':
                df = pd.read_csv(file_path)
            
            print(f"Loaded {dataset_name} from {file_path}")
            print(f"Shape: {df.shape}")
            
            if use_gpu and check_gpu_available():
                return to_gpu(df)
            return df
    
    # If file not found, provide download instructions
    download_url = f"{DATASET_BASE_URL}/{dataset_name}.xls"
    raise FileNotFoundError(
        f"Dataset '{dataset_name}' not found in {data_dir}\n"
        f"Please download from: {download_url}\n"
        f"And save it to: {data_dir}"
    )


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
