# Quick Reference Guide

## Common Functions

### Data Loading
```python
from utils import load_wooldridge_data

df = load_wooldridge_data('wage1')
```

### Running Regressions

**Method 1: Variable lists**
```python
from utils import run_ols, pretty_print_results

results = run_ols(
    data=df,
    dependent='wage',
    independent=['educ', 'exper', 'tenure']
)
pretty_print_results(results)
```

**Method 2: Formula notation**
```python
from utils import run_ols_formula

results = run_ols_formula('wage ~ educ + exper + I(exper**2)', data=df)
```

**With Robust Standard Errors:**
```python
results = run_ols(df, 'wage', ['educ', 'exper'], robust=True)
```

### Diagnostics

**Heteroskedasticity:**
```python
from utils import check_heteroskedasticity
check_heteroskedasticity(results)
```

**Multicollinearity (VIF):**
```python
from utils import check_multicollinearity
check_multicollinearity(df, ['educ', 'exper', 'tenure'])
```

**Autocorrelation:**
```python
from utils import durbin_watson_test
durbin_watson_test(results)
```

### Visualization

**Residual Plots:**
```python
from utils import plot_residuals
plot_residuals(results)
# Or save: plot_residuals(results, save_path='../outputs/residuals.png')
```

**Correlation Matrix:**
```python
from utils import plot_correlation_matrix
plot_correlation_matrix(df)
```

**Summary Statistics:**
```python
from utils import summary_statistics
summary_statistics(df)
```

## Formula Notation Tips

### Transformations
- Square: `I(x**2)`
- Log: `np.log(x)`
- Interaction: `x1:x2`
- Categorical: `C(category)`

### Examples
```python
# Quadratic
'wage ~ educ + exper + I(exper**2)'

# Log-log model
'np.log(wage) ~ np.log(educ) + exper'

# Interaction
'wage ~ educ + exper + educ:exper'

# With categorical
'wage ~ educ + C(industry) + exper'
```

## GPU Acceleration

### Check Availability
```python
from utils import check_gpu_available
print(check_gpu_available())
```

### Bootstrap with GPU
```python
from utils import bootstrap_ols, plot_bootstrap_distribution

results = bootstrap_ols(
    data=df,
    dependent='wage',
    independent=['educ', 'exper'],
    n_iterations=1000,
    use_gpu=True
)

plot_bootstrap_distribution(results, 'educ')
```

## Statsmodels Direct Access

All utility functions return standard statsmodels result objects, so you can also use:

```python
# Standard summary
print(results.summary())

# Access specific values
results.params          # Coefficients
results.bse             # Standard errors
results.tvalues         # t-statistics
results.pvalues         # p-values
results.rsquared        # R-squared
results.rsquared_adj    # Adjusted R-squared
results.fvalue          # F-statistic
results.aic             # AIC
results.bic             # BIC
results.resid           # Residuals
results.fittedvalues    # Fitted values

# Confidence intervals
results.conf_int()

# Predictions
results.predict(new_data)
```

## Common Wooldridge Tests

### Test individual coefficient = 0
```python
# Automatically done in regression output
# Look at t-statistic and p-value
```

### Test multiple restrictions (F-test)
```python
from utils import test_linear_restriction

# Test if two coefficients are equal
test_linear_restriction(results, 'educ = exper')

# Test if multiple coefficients are zero
test_linear_restriction(results, 'educ = 0, exper = 0')
```

### Test for heteroskedasticity
```python
from utils import check_heteroskedasticity
check_heteroskedasticity(results)
# Returns Breusch-Pagan and White test results
```

## File Organization

```
Practice/
├── data/
│   ├── wage1.xls          # Downloaded datasets
│   ├── wage1.txt          # Variable descriptions
│   └── ...
├── notebooks/
│   ├── chapter_02.ipynb   # Your work
│   ├── chapter_03.ipynb
│   └── ...
├── outputs/
│   ├── residuals_ch2.png  # Saved figures
│   └── ...
└── utils/
    └── econometrics_utils.py
```

## Typical Exercise Structure

```python
# 1. Load data
df = load_wooldridge_data('dataset_name')

# 2. Explore
summary_statistics(df)
plot_correlation_matrix(df, ['var1', 'var2', 'var3'])

# 3. Run regression
results = run_ols(df, 'y', ['x1', 'x2', 'x3'])
pretty_print_results(results)

# 4. Diagnostics
check_heteroskedasticity(results)
check_multicollinearity(df, ['x1', 'x2', 'x3'])
plot_residuals(results)

# 5. Interpret and write conclusions
```

## Keyboard Shortcuts (Jupyter)

- `Shift + Enter`: Run cell and move to next
- `Ctrl + Enter`: Run cell and stay
- `A`: Insert cell above
- `B`: Insert cell below
- `M`: Convert to markdown
- `Y`: Convert to code
- `DD`: Delete cell

## Additional Resources

- Statsmodels docs: https://www.statsmodels.org/
- Wooldridge datasets: https://faculty.utrgv.edu/diego.escobari/teaching/Datasets.html
- Markdown guide: https://www.markdownguide.org/basic-syntax/
- Online Textbook: https://drive.google.com/file/d/1Gw_VYjaRxi8Tq-EroKiQLJYuFIW3gs9f/view?pli=1
