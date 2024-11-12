import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, norm
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
import numpy as np
from arch import arch_model
import warnings
from IPython.display import display, Markdown

warnings.filterwarnings("ignore")

# Step 1: Data Retrieval
asset_symbol = "USDCHF=X"  
data = yf.download(asset_symbol, start="2021-01-01", end="2024-09-30")
data['Daily Return'] = data['Close'].pct_change().dropna()

# Asset name extraction for labeling
asset_name = asset_symbol.split('=')[0] if '=' in asset_symbol else asset_symbol

# Step 2: Separate Plots of Closing Price and Daily Returns in (1,2) Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Closing Price Plot
axes[0].plot(data.index, data['Close'], color="red", label=f"{asset_name} Closing Price")
axes[0].set_xlabel("Date")
axes[0].set_ylabel(f"{asset_name} Closing Price")
axes[0].set_title(f"{asset_name} Closing Price")
axes[0].legend()

# Daily Return Plot
axes[1].plot(data.index, data['Daily Return'], color="blue", label=f"R_{asset_name} Daily Return")
axes[1].set_xlabel("Date")
axes[1].set_ylabel(f"R_{asset_name} Daily Return")
axes[1].set_title(f"R_{asset_name} Daily Return")
axes[1].legend()

plt.tight_layout()
plt.show()

# Step 3: Transposed Descriptive Statistics Table for Closing Price and Daily Returns
close_prices = data['Close']
daily_returns = data['Daily Return'].dropna()

def calculate_stats(series, symbol):
    stats = {
        "Financial Instrument Symbol": symbol,
        "Observation Value Start": series.index[0].date(),
        "Observation Value End": series.index[-1].date(),
        "Observation Count": len(series),
        "Mean": series.mean(),
        "Median": series.median(),
        "Max": series.max(),
        "Min": series.min(),
        "Standard Deviation": series.std(),
        "Jarque-Bera Test Statistic": jarque_bera(series)[0],
        "Jarque-Bera p-value": jarque_bera(series)[1],
        "Sum": series.sum(),
        "Total Squared Deviation": np.sum((series - series.mean())**2)
    }
    return pd.DataFrame(stats, index=[symbol])

# Combine and Transpose Descriptive Statistics
desc_stats_df = pd.concat([calculate_stats(close_prices, asset_name), calculate_stats(daily_returns, f"R_{asset_name}")]).T
print(desc_stats_df.to_markdown())

# Step 4: Histogram of Closing Price and Daily Returns with Normal Distribution Line in (1,2) Matrix Format
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histogram of Closing Prices
axes[0].hist(close_prices, bins=50, alpha=0.6, color='red', edgecolor='black', density=True, label=f"{asset_name} Closing Price")
x1 = np.linspace(close_prices.min(), close_prices.max(), 100)
axes[0].plot(x1, norm.pdf(x1, close_prices.mean(), close_prices.std()), 'r--', linewidth=2, label="Normal Dist (Closing Price)")
axes[0].set_xlabel(f"{asset_name} Closing Price")
axes[0].set_ylabel("Density")
axes[0].set_title(f"Histogram of {asset_name} Closing Price with Normal Distribution")
axes[0].legend()

# Histogram of Daily Returns
axes[1].hist(daily_returns, bins=50, alpha=0.6, color='blue', edgecolor='black', density=True, label=f"R_{asset_name} Daily Returns")
x2 = np.linspace(daily_returns.min(), daily_returns.max(), 100)
axes[1].plot(x2, norm.pdf(x2, daily_returns.mean(), daily_returns.std()), 'b--', linewidth=2, label="Normal Dist (Daily Return)")
axes[1].set_xlabel(f"R_{asset_name} Daily Return")
axes[1].set_ylabel("Density")
axes[1].set_title(f"Histogram of R_{asset_name} Daily Returns with Normal Distribution")
axes[1].legend()

plt.tight_layout()
plt.show()

# Step 5: ARCH-LM Test for Daily Return Series
arch_test = het_arch(daily_returns)
arch_test_result = {
    "No": ["**1**"],
    "ARCH-LM Test Statistic": [arch_test[0]],
    "ARCH-LM p-value": [arch_test[1]],
    "Null Hypothesis": ["Reject" if arch_test[1] < 0.05 else "Accept"],
    "Series": [f"R_{asset_name}"]
}
arch_test_df = pd.DataFrame(arch_test_result)
arch_test_df.index = [1]  # Set custom index for readability
print(arch_test_df.to_markdown(index=False))

# Step 6: Fit Various GARCH Models and Display Results in Separate Tables
garch_orders = [(p, q) for p in range(1, 5) for q in range(1, 5)]
results_list = []  # For storing AIC, Q-test, and ARCH-LM results

for p, q in garch_orders:
    model_name = f"GARCH({p},{q})"
    model = arch_model(daily_returns, vol='Garch', p=p, q=q, dist='normal', rescale=True)
    fit = model.fit(disp="off")
    
    params = fit.params
    std_err = np.sqrt(np.diag(fit.param_cov))
    t_vals = params / std_err
    p_vals = 2 * (1 - norm.cdf(np.abs(t_vals)))
    aic = fit.aic

    # Generate dynamic equation with estimated parameters
    omega = params.get("omega", 0)
    alpha_terms = " + ".join([f"{params.get(f'alpha[{i+1}]', 0):.4f} \\epsilon^2_{{t-{i+1}}}" for i in range(p)])
    beta_terms = " + ".join([f"{params.get(f'beta[{i+1}]', 0):.4f} \\sigma^2_{{t-{i+1}}}" for i in range(q)])
    garch_eq = rf"$$ \sigma^2_t = {omega:.4f} + {alpha_terms} + {beta_terms} $$"
    
    # Display model name and equation
    display(Markdown(f"### {model_name} Results for R_{asset_name}"))
    display(Markdown(garch_eq))  # Display the equation with specific model parameters
    
    # Ljung-Box and ARCH-LM Tests
    ljung_box = acorr_ljungbox(fit.resid, lags=[10], return_df=True)
    q_stat, q_pval = ljung_box['lb_stat'].values[0], ljung_box['lb_pvalue'].values[0]
    q_result = "Reject" if q_pval < 0.05 else "Accept"

    arch_test_stat, arch_test_pval, _, _ = het_arch(fit.resid)
    arch_result = "Reject" if arch_test_pval < 0.05 else "Accept"

    # Model Results Dataframe
    model_results = {
        "Parameter": ["omega"] + [f"alpha[{i+1}]" for i in range(p)] + [f"beta[{i+1}]" for i in range(q)] + ["AIC", "Q Stat", "Q p-value", "Q Result", "ARCH LM Stat", "ARCH LM p-value", "ARCH Result"],
        "Value": [params.get("omega", np.nan)] + [params.get(f"alpha[{i+1}]", np.nan) for i in range(p)] + [params.get(f"beta[{i+1}]", np.nan) for i in range(q)] + [aic, q_stat, q_pval, q_result, arch_test_stat, arch_test_pval, arch_result],
        "Standard Error": [std_err[0] if len(std_err) > 0 else np.nan] + [std_err[i+1] for i in range(p)] + [std_err[p+1+i] for i in range(q)] + [np.nan]*7,
        "t-value": [t_vals[0] if len(t_vals) > 0 else np.nan] + [t_vals[i+1] for i in range(p)] + [t_vals[p+1+i] for i in range(q)] + [np.nan]*7,
        "p-value": [p_vals[0] if len(p_vals) > 0 else np.nan] + [p_vals[i+1] for i in range(p)] + [p_vals[p+1+i] for i in range(q)] + [np.nan]*7
    }
    
    # Append to results list
    results_list.append([model_name, aic, q_result, arch_result])
    model_df = pd.DataFrame(model_results)
    display(model_df)

# Step 7: Sort and Display Model Comparison by AIC
results_df = pd.DataFrame(results_list, columns=["Model", "AIC", "Q Result", "ARCH-LM Result"]).sort_values(by="AIC")
display(Markdown("### Model Comparison by AIC"))
print(results_df.to_markdown(index=False))
