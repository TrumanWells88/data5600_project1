
"""
nil_regression_diagnostics.py

Tests the classic linear-regression assumptions -- Linearity, Independence,
Normality, Constant (equal) Variance, and Identifiability -- for a simple
linear model:

    Men's Basketball NIL Estimate ($M)  ~  Rank

fit on `sideline-basketball-nil-rankings.csv`, and produces one chart per
assumption as evidence.

Dependencies: numpy, pandas, matplotlib only (no scipy, no statsmodels).
Every statistic below -- OLS coefficients, Durbin-Watson, Breusch-Pagan,
VIF, normal quantiles for the Q-Q plot, and the Jarque-Bera normality
test -- is computed from first principles using numpy/math.

Usage:
    python3 nil_regression_diagnostics.py path/to/sideline-basketball-nil-rankings.csv
"""

import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESP_COL = "Men's Basketball NIL Estimate ($M)"
PRED_COL = "Rank"

# --------------------------------------------------------------------------
# Small OLS toolkit (replaces statsmodels, which isn't installed)
# --------------------------------------------------------------------------

def ols_fit(X, y):
    """Least squares fit. X must already include an intercept column."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    return beta, fitted, resid


def r_squared(y, fitted):
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def durbin_watson(resid):
    """~2 => no autocorrelation; <<2 => positive autocorrelation."""
    diffs = np.diff(resid)
    return np.sum(diffs ** 2) / np.sum(resid ** 2)


def vif_scores(X, names):
    """Variance Inflation Factor per column (excluding the intercept)."""
    scores = {}
    for i in range(1, X.shape[1]):
        y_i = X[:, i]
        X_others = np.delete(X, i, axis=1)
        _, fitted_i, _ = ols_fit(X_others, y_i)
        r2_i = r_squared(y_i, fitted_i)
        scores[names[i]] = np.inf if r2_i >= 1 else 1 / (1 - r2_i)
    return scores


# --------------------------------------------------------------------------
# Statistics helpers that normally come from scipy.stats -- reimplemented
# here with just the standard-library `math` module and numpy.
# --------------------------------------------------------------------------

def _gamma_series(a, x):
    """Lower regularized incomplete gamma P(a, x) via its series expansion
    (valid for x < a + 1). Numerical Recipes' gser()."""
    max_iter, eps = 500, 3e-16
    ap = a
    summ = 1.0 / a
    delta = summ
    for _ in range(max_iter):
        ap += 1.0
        delta *= x / ap
        summ += delta
        if abs(delta) < abs(summ) * eps:
            break
    return summ * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_cf(a, x):
    """Upper regularized incomplete gamma Q(a, x) via a continued fraction
    (valid for x >= a + 1). Numerical Recipes' gcf()."""
    max_iter, eps, fpmin = 500, 3e-16, 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_cdf(x, df):
    """CDF of a chi-square distribution with `df` degrees of freedom,
    via the regularized incomplete gamma function P(df/2, x/2)."""
    if x <= 0:
        return 0.0
    a, xx = df / 2.0, x / 2.0
    if xx < a + 1.0:
        return _gamma_series(a, xx)
    return 1.0 - _gamma_cf(a, xx)


def norm_ppf(p):
    """Inverse standard-normal CDF (quantile function) via Peter Acklam's
    rational approximation -- accurate to ~1e-9, no scipy required."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def qq_plot(ax, resid):
    """Draws a normal Q-Q plot (matplotlib only) with a reference line
    fit through the sorted residuals vs. theoretical normal quantiles."""
    n = len(resid)
    sorted_resid = np.sort(resid)
    positions = (np.arange(1, n + 1) - 0.5) / n          # Blom-style plotting positions
    theoretical_q = np.array([norm_ppf(p) for p in positions])
    ax.scatter(theoretical_q, sorted_resid, s=14, alpha=0.6)
    slope, intercept = np.polyfit(theoretical_q, sorted_resid, 1)
    line_x = np.array([theoretical_q.min(), theoretical_q.max()])
    ax.plot(line_x, intercept + slope * line_x, color="red", lw=1.5)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Ordered residuals")


def jarque_bera(resid):
    """Jarque-Bera normality test (uses sample skewness & excess kurtosis;
    ~chi2 with 2 df under the null of normality). Standalone replacement
    for scipy.stats.shapiro / scipy.stats.jarque_bera."""
    n = len(resid)
    mean = resid.mean()
    std = resid.std(ddof=0)
    skew = np.mean(((resid - mean) / std) ** 3)
    excess_kurt = np.mean(((resid - mean) / std) ** 4) - 3.0
    jb_stat = (n / 6.0) * (skew ** 2 + (excess_kurt ** 2) / 4.0)
    p_value = 1.0 - chi2_cdf(jb_stat, df=2)
    return jb_stat, p_value


def breusch_pagan(resid, X):
    """Regress squared residuals on the predictors; LM test for heteroskedasticity."""
    n = len(resid)
    resid2 = resid ** 2
    _, fitted_bp, _ = ols_fit(X, resid2)
    r2_bp = r_squared(resid2, fitted_bp)
    lm_stat = n * r2_bp
    dof = X.shape[1] - 1
    p_value = 1.0 - chi2_cdf(lm_stat, dof)
    return lm_stat, p_value


# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------

path = sys.argv[1] if len(sys.argv) > 1 else "sideline-basketball-nil-rankings.csv"
df = pd.read_csv(path, comment="#")
df = df.dropna(subset=[PRED_COL, RESP_COL]).reset_index(drop=True)

x = df[PRED_COL].to_numpy(dtype=float)
y = df[RESP_COL].to_numpy(dtype=float)
n = len(x)

X = np.column_stack([np.ones(n), x])            # simple model: NIL ~ Rank
beta, fitted, resid = ols_fit(X, y)
r2 = r_squared(y, fitted)
std_resid = resid / resid.std(ddof=2)

print(f"n = {n}")
print(f"Fitted model: NIL_est = {beta[0]:.3f} + {beta[1]:.4f} * Rank")
print(f"R^2 = {r2:.3f}")

# --------------------------------------------------------------------------
# 1. LINEARITY
# --------------------------------------------------------------------------
# A linear fit is appropriate only if the residuals show no leftover curved
# pattern once plotted against the fitted values / predictor.

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(x, y, s=12, alpha=0.5)
order = np.argsort(x)
axes[0].plot(x[order], fitted[order], color="red", lw=2, label="OLS fit")
axes[0].set_xlabel("Rank")
axes[0].set_ylabel(RESP_COL)
axes[0].set_title("NIL Estimate vs. Rank (raw)")
axes[0].legend()

axes[1].scatter(fitted, resid, s=12, alpha=0.5)
axes[1].axhline(0, color="red", lw=1.5)
axes[1].set_xlabel("Fitted values")
axes[1].set_ylabel("Residuals")
axes[1].set_title("Residuals vs. Fitted (linearity check)")
fig.tight_layout()
fig.savefig("chart1_linearity.png", dpi=150)
plt.close(fig)

# --------------------------------------------------------------------------
# 2. INDEPENDENCE
# --------------------------------------------------------------------------
# Because Rank is itself an ordering, residuals are plotted in Rank order
# and checked for runs / trends with the Durbin-Watson statistic.

dw = durbin_watson(resid)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x, resid, marker="o", markersize=3, linewidth=0.7, alpha=0.7)
ax.axhline(0, color="red", lw=1.5)
ax.set_xlabel("Rank (observation order)")
ax.set_ylabel("Residual")
ax.set_title(f"Residuals vs. Rank order (independence check)\nDurbin-Watson = {dw:.3f} (2.0 = no autocorrelation)")
fig.tight_layout()
fig.savefig("chart2_independence.png", dpi=150)
plt.close(fig)

# --------------------------------------------------------------------------
# 3. CONSTANT VARIANCE (homoscedasticity)
# --------------------------------------------------------------------------

bp_stat, bp_p = breusch_pagan(resid, X)

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(fitted, np.sqrt(np.abs(std_resid)), s=12, alpha=0.5)
ax.set_xlabel("Fitted values")
ax.set_ylabel("sqrt(|Standardized residual|)")
ax.set_title(
    f"Scale-Location plot (constant-variance check)\n"
    f"Breusch-Pagan LM = {bp_stat:.2f}, p = {bp_p:.2e}"
)
fig.tight_layout()
fig.savefig("chart3_constant_variance.png", dpi=150)
plt.close(fig)

# --------------------------------------------------------------------------
# 4. NORMALITY of residuals
# --------------------------------------------------------------------------

jb_stat, jb_p = jarque_bera(resid)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
qq_plot(axes[0], resid)
axes[0].set_title(f"Normal Q-Q plot\nJarque-Bera = {jb_stat:.2f}, p = {jb_p:.2e}")

axes[1].hist(resid, bins=30, edgecolor="black", alpha=0.75)
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Count")
axes[1].set_title("Histogram of residuals")
fig.tight_layout()
fig.savefig("chart4_normality.png", dpi=150)
plt.close(fig)

# --------------------------------------------------------------------------
# 5. IDENTIFIABILITY
# --------------------------------------------------------------------------
# Identifiability asks whether the design matrix has full column rank (a
# unique OLS solution exists) and how ill-conditioned it is. We test this
# on the simple model AND on an expanded model that adds Tier as dummy
# variables, which is where identifiability problems (the "dummy trap",
# near-collinearity between Rank and Tier) actually tend to show up.

rank_X = np.linalg.matrix_rank(X)
cond_X = np.linalg.cond(X)

tier_dummies = pd.get_dummies(df["Tier"], drop_first=True).astype(float)
names_multi = ["Intercept", "Rank"] + list(tier_dummies.columns)
X_multi = np.column_stack([X, tier_dummies.to_numpy()])
rank_multi = np.linalg.matrix_rank(X_multi)
cond_multi = np.linalg.cond(X_multi)
vifs = vif_scores(X_multi, names_multi)

fig, ax = plt.subplots(figsize=(9, 5))
labels = list(vifs.keys())
values = [vifs[k] for k in labels]
colors = ["red" if v > 5 else "orange" if v > 2.5 else "steelblue" for v in values]
ax.barh(labels, values, color=colors)
ax.axvline(5, color="red", ls="--", lw=1, label="VIF = 5 (concern)")
ax.axvline(10, color="darkred", ls="--", lw=1, label="VIF = 10 (severe)")
ax.set_xlabel("Variance Inflation Factor")
ax.set_title(
    f"Identifiability check: VIF per predictor (Rank + Tier dummies)\n"
    f"design matrix rank = {rank_multi}/{X_multi.shape[1]} columns, "
    f"condition number = {cond_multi:,.0f}"
)
ax.legend()
fig.tight_layout()
fig.savefig("chart5_identifiability.png", dpi=150)
plt.close(fig)

# --------------------------------------------------------------------------
# Summary printed to console
# --------------------------------------------------------------------------

print("\n--- Assumption checks ---")
print(f"Linearity      : R^2 = {r2:.3f}; see chart1 for curvature in residuals")
print(f"Independence   : Durbin-Watson = {dw:.3f} (want close to 2.0)")
print(f"Const. variance: Breusch-Pagan p = {bp_p:.2e} (p < 0.05 => heteroskedastic)")
print(f"Normality      : Jarque-Bera p = {jb_p:.2e} (p < 0.05 => non-normal)")
print(f"Identifiability: simple model rank {rank_X}/{X.shape[1]}, cond# {cond_X:,.1f}; "
      f"multi model rank {rank_multi}/{X_multi.shape[1]}, cond# {cond_multi:,.1f}")
print("VIFs (Rank + Tier dummies):")
for k, v in vifs.items():
    print(f"   {k:15s} {v:8.2f}")

# --------------------------------------------------------------------------
# BONUS: the fix -- log-transform the response and log-transform Rank.
# NIL money vs. rank is a classic power-law / Zipf-type decay, so
# log(NIL) ~ log(Rank) is the natural linearizing transform.
# --------------------------------------------------------------------------

log_x = np.log(x)
log_y = np.log(y)
X_log = np.column_stack([np.ones(n), log_x])
beta_log, fitted_log, resid_log = ols_fit(X_log, log_y)
r2_log = r_squared(log_y, fitted_log)
dw_log = durbin_watson(resid_log)
bp_log_stat, bp_log_p = breusch_pagan(resid_log, X_log)
jb_log_stat, jb_log_p = jarque_bera(resid_log)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].scatter(log_x, log_y, s=12, alpha=0.5)
order_log = np.argsort(log_x)
axes[0, 0].plot(log_x[order_log], fitted_log[order_log], color="red", lw=2)
axes[0, 0].set_xlabel("log(Rank)")
axes[0, 0].set_ylabel("log(NIL Estimate)")
axes[0, 0].set_title(f"log(NIL) vs. log(Rank), R^2 = {r2_log:.3f}")

axes[0, 1].scatter(fitted_log, resid_log, s=12, alpha=0.5)
axes[0, 1].axhline(0, color="red", lw=1.5)
axes[0, 1].set_xlabel("Fitted (log scale)")
axes[0, 1].set_ylabel("Residual (log scale)")
axes[0, 1].set_title(f"Residuals vs. Fitted\nBreusch-Pagan p = {bp_log_p:.3f}")

qq_plot(axes[1, 0], resid_log)
axes[1, 0].set_title(f"Normal Q-Q (log model)\nJarque-Bera p = {jb_log_p:.3f}")

axes[1, 1].plot(x, resid_log, marker="o", markersize=3, linewidth=0.7, alpha=0.7)
axes[1, 1].axhline(0, color="red", lw=1.5)
axes[1, 1].set_xlabel("Rank (observation order)")
axes[1, 1].set_ylabel("Residual (log scale)")
axes[1, 1].set_title(f"Residuals vs. order\nDurbin-Watson = {dw_log:.3f}")

fig.suptitle("After the fix: log(NIL) ~ log(Rank)", fontsize=14)
fig.tight_layout()
fig.savefig("chart6_log_transform_fix.png", dpi=150)
plt.close(fig)

print("\n--- After log-log transform (the proposed fix) ---")
print(f"R^2            = {r2_log:.3f} (vs {r2:.3f} before)")
print(f"Durbin-Watson  = {dw_log:.3f} (vs {dw:.3f} before)")
print(f"Breusch-Pagan p= {bp_log_p:.3f} (vs {bp_p:.2e} before)")
print(f"Jarque-Bera p  = {jb_log_p:.3f} (vs {jb_p:.2e} before)")
