"""Correlation, covariance-based risk decomposition, and factor analysis."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau, norm, pearsonr, spearmanr
import plotly.figure_factory as ff

TRADING_DAYS = 252


def aligned_returns(price_df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Daily returns for the given tickers, restricted to dates where all
    of them have data (required for a well-defined covariance matrix)."""
    return price_df[tickers].pct_change().dropna(how="any")


def correlation_matrix(returns_df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return returns_df.corr(method=method)


def pairwise_stats(returns_df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Pearson r, p-value, and Fisher-z confidence interval for every
    distinct pair of columns."""
    cols = list(returns_df.columns)
    n = len(returns_df)
    z_crit = norm.ppf(1 - (1 - confidence) / 2)
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            r, p = pearsonr(returns_df[a], returns_df[b])
            r_clamped = min(max(r, -0.999999), 0.999999)
            z = np.arctanh(r_clamped)
            se = 1 / np.sqrt(n - 3) if n > 3 else np.nan
            lo, hi = np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)
            rows.append({
                "Pair": f"{a} / {b}", "Pearson r": r, "p-value": p,
                "CI Low": lo, "CI High": hi, "N": n,
                "Significant": p < (1 - confidence),
            })
    return pd.DataFrame(rows)


def rolling_correlation(returns_df: pd.DataFrame, a: str, b: str, window: int) -> pd.Series:
    return returns_df[a].rolling(window).corr(returns_df[b])


def risk_decomposition(returns_df: pd.DataFrame, weights: dict):
    """Covariance-based marginal/component contribution to portfolio risk.
    Returns (table, portfolio_annualized_vol). Component contributions sum
    exactly to the portfolio volatility (Euler decomposition)."""
    tickers = list(weights.keys())
    R = returns_df[tickers]
    cov_annual = R.cov().values * TRADING_DAYS
    w = np.array([weights[t] for t in tickers], dtype=float)

    port_var = float(w @ cov_annual @ w)
    port_vol = float(np.sqrt(port_var)) if port_var > 0 else np.nan

    standalone_vol = np.sqrt(np.diag(cov_annual))
    if port_vol and not np.isnan(port_vol):
        mctr = (cov_annual @ w) / port_vol
    else:
        mctr = np.full(len(w), np.nan)
    cctr = w * mctr
    pct_ctr = cctr / port_vol * 100 if port_vol else np.full(len(w), np.nan)

    table = pd.DataFrame({
        "Weight": w,
        "Standalone Ann. Vol": standalone_vol,
        "MCTR": mctr,
        "CCTR (Ann. Vol)": cctr,
        "% of Portfolio Risk": pct_ctr,
    }, index=tickers)
    return table, port_vol


def diversification_ratio(risk_table: pd.DataFrame, port_vol: float) -> float:
    gross_weighted_vol = float((risk_table["Weight"].abs() * risk_table["Standalone Ann. Vol"]).sum())
    return gross_weighted_vol / port_vol if port_vol else np.nan


def conditional_correlation(returns_df: pd.DataFrame, reference_returns: pd.Series,
                             quantile: float = 0.10, method: str = "pearson"):
    """Correlation matrix restricted to the worst `quantile` fraction of days
    (ranked by reference_returns), vs. the full-sample matrix."""
    ref = reference_returns.reindex(returns_df.index).dropna()
    threshold = ref.quantile(quantile)
    stress_dates = ref[ref <= threshold].index
    stress_dates = returns_df.index.intersection(stress_dates)
    full = returns_df.corr(method=method)
    stress = returns_df.loc[stress_dates].corr(method=method)
    return full, stress, len(stress_dates), float(threshold)


def pca_decomposition(corr_df: pd.DataFrame):
    """Eigen-decomposition of a correlation matrix. Returns
    (explained_variance_ratio, loadings_df, eigenvalues), sorted descending."""
    vals, vecs = np.linalg.eigh(corr_df.values)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    explained = vals / vals.sum()
    loadings = pd.DataFrame(vecs, index=corr_df.index,
                             columns=[f"PC{i+1}" for i in range(len(vals))])
    return explained, loadings, vals


def dendrogram_figure(corr_df: pd.DataFrame, colors=None):
    """Hierarchical clustering dendrogram using the standard correlation
    distance d = sqrt(0.5 * (1 - rho)), average linkage."""
    dist = np.sqrt(np.clip(0.5 * (1 - corr_df.values), 0, None))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    fig = ff.create_dendrogram(
        corr_df.values, labels=list(corr_df.columns),
        distfun=lambda x: condensed,
        linkagefun=lambda x: linkage(x, method="average"),
        colorscale=colors,
    )
    return fig
