import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

df_log = pd.read_csv(
    "cfn_volumencreditocontingente_cleaned.csv",
)
print(df_log)

# Formula uses 'log_monto' as target
target = "log_monto"
predictors = [c for c in df_log.columns if c != target]
formula_str = f"{target} ~ {' + '.join(predictors)}"
print(formula_str)

result_ln = smf.ols(formula=formula_str, data=df_log).fit()
result_summary = result_ln.summary()
# print(result_summary)

tab_coeff = pd.DataFrame(
    result_summary.tables[1].data[1:],
    columns=result_summary.tables[1].data[0],
)
tab_coeff.columns = ["Variable", "coef", "std err", "t", "P>|t|", "[0.025]", "[0.975]"]

tab_coeff["Variable"] = np.where(
    tab_coeff["Variable"].str.len() < 32,
    tab_coeff["Variable"],
    tab_coeff["Variable"].str[:30] + "..",
)

tab_coeff["exp(coef)"] = np.exp(tab_coeff["coef"].astype(float))

tab_coeff["Variable"] = tab_coeff["Variable"].str.replace("_", "\_", regex=False)

tab_coeff.to_csv("regression_summary_pandas.csv", index=False)

# Generate longtable LaTeX code
latex_code = tab_coeff.to_latex(
    index=False,
    float_format="%.2f",
    # longtable=False,
    # caption="My Long Table",
    # label="tab:model_summary",
)
with open("regression_summary2.txt", "w") as f:
    f.write(latex_code)

# Save summary to txt file
with open("regression_summary1.txt", "w") as f:
    f.write(result_ln.summary().as_text())

# Extract residuals and fitted values
resid = result_ln.resid
fitted = result_ln.fittedvalues

# ---- Supuesto 1: Linealidad ----
print("\n=== LINEALIDAD ===")
print(f"R-squared: {result_ln.rsquared:.4f}")

# ---- Plot 1: Residuals vs Fitted ----
# plt.figure(figsize=(7, 5))
# sns.scatterplot(x=fitted, y=resid, alpha=0.5)
# plt.axhline(0, color="r", linestyle="--")
# plt.title("Residuals vs Fitted (Log Scale)")
# plt.xlabel("Predicted Log Amount")
# plt.ylabel("Residuals")
# plt.tight_layout()
# plt.savefig("residuals_vs_fitted.png")  # Save the plot as an image file
# plt.show()

# ---- Supuesto 2: Normalidad de residuos ----
print("\n=== NORMALIDAD ===")
from scipy.stats import shapiro, kstest

# Shapiro-Wilk test
stat, p_value = shapiro(resid)
print(f"Shapiro-Wilk test: statistic={stat:.4f}, p-value={p_value:.6f}")

# Kolmogorov-Smirnov test
ks_stat, ks_pval = kstest(resid, "norm", args=(resid.mean(), resid.std()))
print(f"Kolmogorov-Smirnov test: statistic={ks_stat:.4f}, p-value={ks_pval:.6f}")

# ---- Plot 2: Q-Q Plot (Normality Check) ----
# plt.figure(figsize=(7, 5))
# sm.qqplot(resid, line="45", fit=True)
# plt.title("Q-Q Plot (Normality Check)")
# plt.tight_layout()
# plt.savefig("qq_plot.png")  # Save the plot as an image file
# plt.show()

# ---- Plot 3: Histogram de residuos ----
# plt.figure(figsize=(7, 5))
# plt.hist(resid, bins=30, edgecolor="black", alpha=0.7)
# plt.xlabel("Residuos")
# plt.ylabel("Frecuencia")
# plt.title("Distribución de Residuos")
# plt.tight_layout()
# plt.savefig("residuals_histogram.png")  # Save the plot as an image file
# plt.show()

# ---- Supuesto 3: Homocedasticidad ----
print("\n=== HOMOCEDASTICIDAD ===")
# Breusch-Pagan test
from statsmodels.stats.diagnostic import het_breuschpagan

bp_stat, bp_pval, _, _ = het_breuschpagan(resid, result_ln.model.exog)
print(f"Breusch-Pagan test: statistic={bp_stat:.4f}, p-value={bp_pval:.6f}")

# ---- Plot 4: Scale-Location Plot ----
# plt.figure(figsize=(7, 5))
# standardized_resid = resid / np.std(resid)
# plt.scatter(fitted, np.sqrt(np.abs(standardized_resid)), alpha=0.5)
# plt.xlabel("Fitted values")
# plt.ylabel("√|Standardized residuals|")
# plt.title("Scale-Location Plot")
# plt.tight_layout()
# plt.savefig("scale_location_plot.png")  # Save the plot as an image file
# plt.show()

# ---- Supuesto 4: No colinealidad ----
print("\n=== MULTICOLINEALIDAD ===")
# Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Variable"] = result_ln.model.exog_names[1:]  # Excluir intercepto
vif_data["VIF"] = [
    variance_inflation_factor(result_ln.model.exog, i)
    for i in range(1, result_ln.model.exog.shape[1])
]
print(vif_data)

# ---- Supuesto 5: Independencia de residuos ----
print("\n=== INDEPENDENCIA (Durbin-Watson) ===")
from statsmodels.stats.stattools import durbin_watson

dw_stat = durbin_watson(resid)
print(f"Durbin-Watson statistic: {dw_stat:.4f}")
print("(Valor cercano a 2 indica independencia)")
