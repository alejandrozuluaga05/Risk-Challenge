"""Scenario stress testing (VaR/SVaR), optimal hedge ratio, and benchmark
comparison — single-factor (market model) methodology.

Core model: regress portfolio daily returns on a market-factor proxy
(SPY) to get beta/alpha/R^2 and residual (idiosyncratic) volatility. A
scenario shock S to the market factor implies a systematic portfolio P&L
of beta * S; residual volatility around that shocked mean gives a
closed-form (Gaussian) Stressed VaR / Stressed CVaR. All formulas are
verified against Monte Carlo simulation and brute-force optimization in
the test/validation scripts used during development.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

TRADING_DAYS = 252


def market_model(port_returns: pd.Series, market_returns: pd.Series):
    """OLS regression of portfolio returns on market returns. Returns a
    dict with beta, alpha (daily), r_squared, residual_std (daily), and
    the residual series itself."""
    aligned = pd.concat([port_returns.rename("p"), market_returns.rename("m")],
                         axis=1).dropna()
    beta, alpha, r, p_value, se = stats.linregress(aligned["m"], aligned["p"])
    resid = aligned["p"] - (alpha + beta * aligned["m"])
    resid_std = resid.std(ddof=2) if len(resid) > 2 else np.nan
    return {
        "beta": float(beta), "alpha_daily": float(alpha), "r_squared": float(r ** 2),
        "p_value": float(p_value), "residual_std": float(resid_std),
        "residuals": resid, "n": len(aligned),
    }


def asset_betas(returns_df: pd.DataFrame, market_returns: pd.Series, tickers: list) -> dict:
    """Individual beta of each ticker to the market factor."""
    betas = {}
    for t in tickers:
        aligned = pd.concat([returns_df[t].rename("a"), market_returns.rename("m")],
                             axis=1).dropna()
        beta, alpha, r, p_value, se = stats.linregress(aligned["m"], aligned["a"])
        betas[t] = float(beta)
    return betas


def scenario_shock(beta_p: float, market_shock: float) -> float:
    """Point-estimate systematic P&L from a market factor shock."""
    return beta_p * market_shock


def stressed_var_cvar(beta_p: float, market_shock: float, residual_std: float,
                       confidence: float = 0.95):
    """Closed-form Gaussian Stressed VaR / Stressed CVaR: the portfolio's
    residual (idiosyncratic) distribution re-centered at the scenario's
    systematic shock. Verified against 2M-draw Monte Carlo simulation."""
    mu = beta_p * market_shock
    z = norm.ppf(1 - confidence)
    svar = -(mu + z * residual_std)
    scvar = -(mu - residual_std * norm.pdf(z) / (1 - confidence))
    return float(svar), float(scvar), float(mu)


def optimal_hedge_ratio(port_returns: pd.Series, hedge_returns: pd.Series):
    """Minimum-variance hedge ratio h* = -Cov(p,h)/Var(h) (Ederington 1979).
    Adding h* units of the hedge return to the portfolio return minimizes
    variance. Verified against brute-force grid search."""
    aligned = pd.concat([port_returns.rename("p"), hedge_returns.rename("h")],
                         axis=1).dropna()
    cov = aligned["p"].cov(aligned["h"])
    var_h = aligned["h"].var()
    h_star = -cov / var_h if var_h else np.nan
    rho = aligned["p"].corr(aligned["h"])
    var_unhedged = aligned["p"].var()
    var_at_optimal = var_unhedged * (1 - rho ** 2)
    return {
        "h_star": float(h_star), "rho": float(rho), "hedge_effectiveness": float(rho ** 2),
        "var_unhedged": float(var_unhedged), "var_at_optimal": float(var_at_optimal),
        "vol_unhedged": float(np.sqrt(var_unhedged * TRADING_DAYS)),
        "vol_at_optimal": float(np.sqrt(var_at_optimal * TRADING_DAYS)),
    }


def hedged_variance(port_returns: pd.Series, hedge_returns: pd.Series, h: float) -> float:
    aligned = pd.concat([port_returns.rename("p"), hedge_returns.rename("h")],
                         axis=1).dropna()
    return float((aligned["p"] + h * aligned["h"]).var())


def variance_curve(port_returns: pd.Series, hedge_returns: pd.Series, h_range=(-1.0, 1.0), n=201):
    hs = np.linspace(h_range[0], h_range[1], n)
    variances = [hedged_variance(port_returns, hedge_returns, h) for h in hs]
    return hs, np.array(variances)


def benchmark_stats(port_returns: pd.Series, bench_returns: pd.Series):
    """Alpha/beta/R^2, tracking error, information ratio, and up/down
    capture ratios of a portfolio vs. a benchmark return series."""
    aligned = pd.concat([port_returns.rename("p"), bench_returns.rename("b")],
                         axis=1).dropna()
    beta, alpha_daily, r, p_value, se = stats.linregress(aligned["b"], aligned["p"])
    alpha_annual = alpha_daily * TRADING_DAYS

    diff = aligned["p"] - aligned["b"]
    te_annual = diff.std(ddof=1) * np.sqrt(TRADING_DAYS)
    ir = (diff.mean() * TRADING_DAYS) / te_annual if te_annual else np.nan

    up = aligned["b"] > 0
    down = aligned["b"] < 0
    up_capture = (aligned["p"][up].mean() / aligned["b"][up].mean()) if up.any() else np.nan
    down_capture = (aligned["p"][down].mean() / aligned["b"][down].mean()) if down.any() else np.nan

    return {
        "beta": float(beta), "alpha_annual": float(alpha_annual), "r_squared": float(r ** 2),
        "tracking_error": float(te_annual), "information_ratio": float(ir),
        "up_capture": float(up_capture), "down_capture": float(down_capture), "n": len(aligned),
    }
