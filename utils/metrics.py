"""Portfolio performance and risk metric calculations."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def portfolio_daily_returns(price_df: pd.DataFrame, weights: dict) -> pd.Series:
    """Fixed-weight (daily-rebalanced) portfolio return series from an
    adjusted-close price DataFrame (columns = tickers). Weights are used
    exactly as given (not renormalized), so negative weights for short
    positions and net exposure other than 100% are both valid."""
    rets = price_df.pct_change().dropna(how="all")
    w = pd.Series(weights, dtype=float)
    aligned = rets[w.index].dropna(how="any")
    return (aligned * w).sum(axis=1).rename("portfolio")


def net_exposure(weights: dict) -> float:
    return float(sum(weights.values()))


def gross_exposure(weights: dict) -> float:
    return float(sum(abs(w) for w in weights.values()))


def equity_curve(returns: pd.Series, base: float = 100.0) -> pd.Series:
    return base * (1 + returns).cumprod()


def cumulative_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def cagr(returns: pd.Series) -> float:
    if len(returns) == 0:
        return np.nan
    total = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    if years <= 0:
        return np.nan
    return float(total ** (1 / years) - 1)


def ytd_return(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    current_year = returns.index.max().year
    ytd = returns[returns.index.year == current_year]
    return cumulative_return(ytd)


def annualized_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    if returns.std() == 0 or returns.empty:
        return np.nan
    rf_daily = risk_free_annual / TRADING_DAYS
    excess = returns - rf_daily
    return float(excess.mean() / returns.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    curve = equity_curve(returns, base=1.0)
    running_max = curve.cummax()
    drawdown = curve / running_max - 1
    return float(drawdown.min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    curve = equity_curve(returns, base=1.0)
    running_max = curve.cummax()
    return curve / running_max - 1


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR as a positive loss fraction (e.g. 0.03 = 3% loss)."""
    if returns.empty:
        return np.nan
    return float(-np.percentile(returns, (1 - confidence) * 100))


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical CVaR (Expected Shortfall) as a positive loss fraction."""
    if returns.empty:
        return np.nan
    var_threshold = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= var_threshold]
    if tail.empty:
        return float(-var_threshold)
    return float(-tail.mean())


def summary_metrics(returns: pd.Series, risk_free_annual: float = 0.0,
                     confidence: float = 0.95) -> dict:
    return {
        "Total Return": cumulative_return(returns),
        "YTD Return": ytd_return(returns),
        "CAGR": cagr(returns),
        "Ann. Volatility": annualized_vol(returns),
        "Sharpe Ratio": sharpe_ratio(returns, risk_free_annual),
        "Max Drawdown": max_drawdown(returns),
        f"VaR {int(confidence*100)}% (1d)": historical_var(returns, confidence),
        f"CVaR {int(confidence*100)}% (1d)": historical_cvar(returns, confidence),
    }


HORIZONS = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}


def trailing_slice(returns: pd.Series, years: float):
    """Trailing N-year slice of a return series ending at its last date.
    Returns None if the series doesn't have roughly that much history."""
    if returns.empty:
        return None
    end = returns.index.max()
    start_target = end - pd.DateOffset(years=years)
    if returns.index.min() > start_target + pd.Timedelta(days=45):
        return None
    return returns[returns.index >= start_target]


def multi_horizon_table(returns: pd.Series, risk_free_annual: float = 0.0,
                         confidence: float = 0.95) -> pd.DataFrame:
    """Total Return / CAGR / Vol / Sharpe / Max DD / VaR / CVaR for each of
    1Y, 3Y, 5Y, 10Y trailing windows. Columns without enough history are NaN."""
    metric_names = ["Total Return", "CAGR", "Ann. Volatility", "Sharpe Ratio",
                     "Max Drawdown", f"VaR {int(confidence*100)}% (1d)",
                     f"CVaR {int(confidence*100)}% (1d)"]
    rows = {}
    for label, years in HORIZONS.items():
        window = trailing_slice(returns, years)
        if window is None or len(window) < 20:
            rows[label] = {k: np.nan for k in metric_names}
        else:
            m = summary_metrics(window, risk_free_annual, confidence)
            m.pop("YTD Return", None)
            rows[label] = m
    return pd.DataFrame(rows)[list(HORIZONS)]
