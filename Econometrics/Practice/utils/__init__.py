"""
Utility package for Wooldridge Econometrics exercises
"""

from .econometrics_utils import (
    # Data loading
    load_wooldridge_data,
    download_wooldridge_data,
    load_wooldridge_data_with_download,
    get_dataset_info,
    
    # Regression
    run_ols,
    run_ols_formula,
    pretty_print_results,
    
    # Diagnostics
    check_heteroskedasticity,
    check_multicollinearity,
    durbin_watson_test,
    
    # Visualization
    plot_residuals,
    plot_correlation_matrix,
    summary_statistics,
    
    # Hypothesis testing
    test_linear_restriction,
    
    # GPU utilities
    check_gpu_available,
    bootstrap_ols,
    plot_bootstrap_distribution,
)

__all__ = [
    'load_wooldridge_data',
    'download_wooldridge_data',
    'load_wooldridge_data_with_download',
    'get_dataset_info',
    'run_ols',
    'run_ols_formula',
    'pretty_print_results',
    'check_heteroskedasticity',
    'check_multicollinearity',
    'durbin_watson_test',
    'plot_residuals',
    'plot_correlation_matrix',
    'summary_statistics',
    'test_linear_restriction',
    'check_gpu_available',
    'bootstrap_ols',
    'plot_bootstrap_distribution',
]
