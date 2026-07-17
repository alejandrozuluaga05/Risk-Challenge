"""Portfolio risk & performance dashboard, powered by Yahoo Finance (yfinance)."""
import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.correlation import (
    aligned_returns, conditional_correlation, correlation_matrix,
    dendrogram_figure, diversification_ratio, pairwise_stats,
    pca_decomposition, risk_decomposition, rolling_correlation,
)
from utils.data import fetch_display_name, fetch_news, fetch_prices, fetch_quote
from utils.metrics import (
    CHART_WINDOW_LABELS, drawdown_series, equity_curve, gross_exposure,
    historical_cvar, historical_var, max_drawdown, multi_horizon_table,
    net_exposure, portfolio_daily_returns, resolve_window, summary_metrics,
)
from utils.scenarios import (
    asset_betas, benchmark_stats, hedged_variance, market_model,
    optimal_hedge_ratio, stressed_var_cvar, variance_curve,
)

NAVY, GREEN, RED, AMBER = "#1B2A4A", "#16a34a", "#dc2626", "#b45309"

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide",
                    page_icon="📊")

st.markdown("""
<style>
.badge { display:inline-block; padding:2px 10px; border-radius:999px;
  font-size:0.75rem; font-weight:600; letter-spacing:.02em; }
.badge-long { background:rgba(22,163,74,0.15); color:#16a34a;
  border:1px solid rgba(22,163,74,0.4); }
.badge-short { background:rgba(220,38,38,0.15); color:#dc2626;
  border:1px solid rgba(220,38,38,0.4); }
.hedge-tag { font-size:0.65rem; font-weight:700; color:#5B6B82; text-transform:uppercase;
  letter-spacing:.04em; }
.badge-sep { color:#C9D2DC; font-weight:400; }
.news-item { padding:10px 2px; border-bottom:1px solid rgba(27,42,74,0.08); }
.news-item:last-child { border-bottom:none; }
.news-title { font-weight:600; text-decoration:none; color:#1B2A4A; transition:color .15s ease; }
.news-title:hover { color:#3F6C9C; text-decoration:underline; }
.news-meta { font-size:0.78rem; opacity:0.65; margin-top:2px; }
.news-proxy-note { font-size:0.78rem; font-style:italic; color:#5B6B82;
  margin-bottom:10px; padding-bottom:8px; border-bottom:1px dashed rgba(27,42,74,0.15); }
.metric-label { font-size:0.82rem; color:#5B6B82; margin-top:2px; }
.metric-value { font-size:1.9rem; font-weight:700; line-height:1.3; margin-bottom:6px;
  font-family: Cambria, Georgia, serif; }
h5 { margin-top: 0.75rem; letter-spacing:.01em; }
div[data-testid="stVerticalBlockBorderWrapper"] {
  box-shadow: 0 1px 5px rgba(15,23,42,0.07); border-radius: 10px;
}
div[data-baseweb="tab-list"] { gap: 4px; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------ session state --
def _init_state():
    if "holdings_meta" not in st.session_state:
        st.session_state.holdings_meta = [
            {"id": "h1", "ticker": "AVGO", "default_weight": 33.34, "default_dir": "Long"},
            {"id": "h2", "ticker": "HGZ26.CMX", "default_weight": 33.33, "default_dir": "Long"},
            {"id": "h3", "ticker": "ZN=F", "default_weight": 33.33, "default_dir": "Short"},
        ]
    if "hedges_meta" not in st.session_state:
        st.session_state.hedges_meta = [
            {"id": "g1", "ticker": "SOXX", "default_weight": 10.0, "default_dir": "Short"},
        ]
    st.session_state.setdefault("next_id", 100)


_init_state()


def signed_weight(item: dict) -> float:
    w = st.session_state.get(f"w_{item['id']}", item["default_weight"]) or 0.0
    d = st.session_state.get(f"d_{item['id']}", item["default_dir"])
    return (1 if d == "Long" else -1) * w / 100


risk_free = st.session_state.get("risk_free", 4.0) / 100
confidence = st.session_state.get("confidence_pct", 95) / 100

holdings = st.session_state.holdings_meta
hedges = st.session_state.hedges_meta
base_weights_raw = {h["ticker"]: signed_weight(h) for h in holdings}
hedge_weights_raw = {}
for g in hedges:
    hedge_weights_raw[g["ticker"]] = hedge_weights_raw.get(g["ticker"], 0) + signed_weight(g)

all_tickers = tuple(sorted(set(base_weights_raw) | set(hedge_weights_raw)))
if not all_tickers:
    st.warning("Add at least one holding in the Controls tab to get started.")
    st.stop()

with st.spinner(f"Pulling {len(all_tickers)} tickers from Yahoo Finance..."):
    try:
        prices = fetch_prices(all_tickers, period="max")
    except Exception as e:
        st.error(f"Failed to fetch data from Yahoo Finance: {e}")
        st.stop()

missing = [t for t in all_tickers if t not in prices.columns or prices[t].dropna().empty]
if missing:
    st.warning(f"No data returned for: {', '.join(missing)}. Excluded below — "
               f"check the ticker symbol(s) in the Controls tab.")

base_weights = {t: w for t, w in base_weights_raw.items() if t not in missing}
hedge_weights_input = {t: w for t, w in hedge_weights_raw.items() if t not in missing}

full_weights = dict(base_weights)
for t, w in hedge_weights_input.items():
    full_weights[t] = full_weights.get(t, 0) + w


# ---------------------------------------------------------------- helpers --
def badge(direction: str) -> str:
    cls = "badge-long" if direction == "Long" else "badge-short"
    return f'<span class="badge {cls}">{direction}</span>'


def render_headlines(ticker: str):
    result = fetch_news(ticker)
    items, proxy = result["items"], result["proxy"]
    if not items:
        st.caption("No recent headlines available for this instrument.")
        return
    if proxy:
        proxy_name = html.escape(fetch_display_name(proxy))
        st.markdown(
            f'<div class="news-proxy-note">{ticker} has no native Yahoo Finance news feed — '
            f'showing related market news via {proxy_name} ({proxy}) instead.</div>',
            unsafe_allow_html=True,
        )
    for item in items:
        try:
            pub_fmt = pd.to_datetime(item["published"]).strftime("%b %d, %Y %H:%M UTC")
        except Exception:
            pub_fmt = ""
        title = html.escape(item["title"] or "(untitled)")
        publisher = html.escape(item["publisher"] or "")
        if item["url"]:
            title_html = f'<a class="news-title" href="{html.escape(item["url"])}" target="_blank">{title}</a>'
        else:
            title_html = f'<span class="news-title">{title}</span>'
        st.markdown(
            f'<div class="news-item">{title_html}'
            f'<div class="news-meta">{publisher} · {pub_fmt}</div></div>',
            unsafe_allow_html=True,
        )


def format_horizon_table(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy().astype(object)
    for idx in table.index:
        for col in table.columns:
            v = table.loc[idx, col]
            if pd.isna(v):
                out.loc[idx, col] = "N/A"
            elif idx == "Sharpe Ratio":
                out.loc[idx, col] = f"{v:.2f}"
            else:
                out.loc[idx, col] = f"{v:.2%}"
    return out


def render_range_selector(key_prefix: str) -> str:
    return st.radio("Time range", CHART_WINDOW_LABELS, index=len(CHART_WINDOW_LABELS) - 1,
                     horizontal=True, key=f"{key_prefix}_range")


def render_charts(window_returns: pd.Series, label: str, key_prefix: str):
    curve = equity_curve(window_returns)
    dd = drawdown_series(window_returns)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve.index, y=curve, line=dict(color="#2563eb", width=2),
                                  fill="tozeroy", fillcolor="rgba(37,99,235,0.06)"))
        fig.update_layout(title=f"Equity Curve ({label}, base = 100)", height=340,
                           margin=dict(t=40, b=20), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_eq")
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(color="#dc2626", width=1)))
        fig.update_layout(title=f"Drawdown ({label})", height=340, yaxis_tickformat=".0%",
                           margin=dict(t=40, b=20), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_dd")


def render_self_contained_charts(returns: pd.Series, key_prefix: str):
    """Range selector + charts, both driven by the same widget (used where
    nothing else on the page needs to react to the selected range)."""
    label = render_range_selector(key_prefix)
    window_returns, capped = resolve_window(returns, label)
    if capped:
        st.info(f"Not enough history for a {label} window; showing full history instead.")
    render_charts(window_returns, label, key_prefix)


def render_ticker_tab(ticker: str):
    if ticker in missing:
        st.error(f"No price data available for **{ticker}**. Check the symbol in the Controls tab.")
        return

    name = fetch_display_name(ticker)
    quote = fetch_quote(ticker)
    returns = prices[ticker].pct_change().dropna()

    h1, h2 = st.columns([3, 1])
    with h1:
        st.subheader(name)
        st.caption(ticker)
    with h2:
        if quote["last"] is not None:
            st.metric("Last Price", f"{quote['last']:.2f}",
                       f"{quote['change_pct']:.2%}" if quote["change_pct"] is not None else None)

    st.markdown("##### Headlines")
    with st.container(border=True):
        render_headlines(ticker)

    st.markdown("##### Performance by Horizon")
    with st.container(border=True):
        st.dataframe(format_horizon_table(multi_horizon_table(returns, risk_free, confidence)),
                     use_container_width=True)

    st.markdown("##### Charts")
    with st.container(border=True):
        render_self_contained_charts(returns, key_prefix=f"tk_{ticker}")


def _metric_cell(col, label: str, value, fmt: str, color: str):
    text = "N/A" if pd.isna(value) else format(value, fmt)
    col.markdown(
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color if not pd.isna(value) else NAVY}">{text}</div>',
        unsafe_allow_html=True,
    )


def render_metric_cards(m: dict):
    var_key = f"VaR {int(confidence*100)}% (1d)"
    cvar_key = f"CVaR {int(confidence*100)}% (1d)"

    row1 = st.columns(4)
    _metric_cell(row1[0], "Total Return", m["Total Return"], ".1%",
                 GREEN if m["Total Return"] >= 0 else RED)
    _metric_cell(row1[1], "YTD Return", m["YTD Return"], ".1%",
                 GREEN if m["YTD Return"] >= 0 else RED)
    _metric_cell(row1[2], "CAGR", m["CAGR"], ".1%", GREEN if m["CAGR"] >= 0 else RED)
    _metric_cell(row1[3], "Sharpe Ratio", m["Sharpe Ratio"], ".2f",
                 GREEN if m["Sharpe Ratio"] >= 0 else RED)

    row2 = st.columns(4)
    _metric_cell(row2[0], "Ann. Volatility", m["Ann. Volatility"], ".1%", NAVY)
    _metric_cell(row2[1], "Max Drawdown", m["Max Drawdown"], ".1%", RED)
    _metric_cell(row2[2], var_key, m[var_key], ".2%", AMBER)
    _metric_cell(row2[3], cvar_key, m[cvar_key], ".2%", AMBER)


def render_portfolio_tab():
    if not base_weights:
        st.warning("No valid holdings to compute a portfolio. Fix tickers in the Controls tab.")
        return

    base_returns = portfolio_daily_returns(prices[list(base_weights)], base_weights)

    st.title("Portfolio")
    st.caption(f"Data source: Yahoo Finance (yfinance) · {base_returns.index.min().date()} → "
               f"{base_returns.index.max().date()} · {len(base_returns)} trading days")

    holdings_badges = " &nbsp; ".join(
        f'{t} {abs(w):.1%} {badge("Long" if w >= 0 else "Short")}'
        for t, w in base_weights.items()
    )
    badge_line = holdings_badges
    if hedge_weights_input:
        hedge_badges = " &nbsp; ".join(
            f'{t} {abs(w):.1%} {badge("Long" if w >= 0 else "Short")} '
            f'<span class="hedge-tag">Hedge</span>'
            for t, w in hedge_weights_input.items()
        )
        badge_line += '&nbsp; <span class="badge-sep">│</span> &nbsp;' + hedge_badges
    st.markdown(badge_line, unsafe_allow_html=True)
    st.caption(f"Net exposure: {net_exposure(base_weights):.0%} · "
               f"Gross exposure: {gross_exposure(base_weights):.0%}")

    st.markdown("##### Holdings Allocation")
    with st.container(border=True):
        pie_tickers = list(base_weights.keys())
        pie_values = [abs(base_weights[t]) for t in pie_tickers]
        pie_colors = ["#16a34a" if base_weights[t] >= 0 else "#dc2626" for t in pie_tickers]
        fig = go.Figure(data=[go.Pie(
            labels=pie_tickers, values=pie_values, marker=dict(colors=pie_colors,
            line=dict(color="#FFFFFF", width=2)),
            hole=0.45, textinfo="label+percent", texttemplate="%{label}<br>%{percent}",
            textfont=dict(size=13),
        )])
        fig.update_layout(height=380, margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key="holdings_pie")
        st.caption("Slice size = gross weight (magnitude); color indicates direction "
                   "(green = long, red = short). Hedge instruments are shown separately above.")

    st.markdown("##### Headlines (by holding)")
    with st.container(border=True):
        news_tabs = st.tabs(list(base_weights))
        for t, nt in zip(base_weights, news_tabs):
            with nt:
                render_headlines(t)

    st.markdown("##### Performance by Horizon")
    with st.container(border=True):
        st.dataframe(format_horizon_table(multi_horizon_table(base_returns, risk_free, confidence)),
                     use_container_width=True)

    st.markdown("##### Time Range")
    range_label = render_range_selector("portfolio")
    window_returns, capped = resolve_window(base_returns, range_label)
    if capped:
        st.info(f"Not enough history for a {range_label} window; showing full history instead.")

    st.markdown(f"##### Key Metrics ({range_label})")
    with st.container(border=True):
        render_metric_cards(summary_metrics(window_returns, risk_free, confidence))

    st.markdown("##### Charts")
    with st.container(border=True):
        render_charts(window_returns, range_label, key_prefix="portfolio")

    if not hedge_weights_input:
        st.info("Add a hedge instrument in the Controls tab to compare pre- and "
                "post-hedge return distributions and risk metrics.")
        return

    st.divider()
    st.header("Hedged Portfolio Comparison")

    hedged_weights = dict(base_weights)
    for t, w in hedge_weights_input.items():
        hedged_weights[t] = hedged_weights.get(t, 0) + w

    hedged_returns = portfolio_daily_returns(prices[list(hedged_weights)], hedged_weights)

    st.markdown(
        "Hedge overlay added on top of the base weights: " +
        ", ".join(f'{t} {abs(w):.0%} {badge("Long" if w >= 0 else "Short")}'
                   for t, w in hedge_weights_input.items()),
        unsafe_allow_html=True,
    )
    st.caption(f"Hedged sleeve — Net exposure: {net_exposure(hedged_weights):.0%} · "
               f"Gross exposure: {gross_exposure(hedged_weights):.0%}")

    st.markdown("##### Metrics: Before vs After Hedge")
    with st.container(border=True):
        base_metrics = summary_metrics(base_returns, risk_free, confidence)
        hedged_metrics = summary_metrics(hedged_returns, risk_free, confidence)
        metric_rows = [{"Metric": k, "Before Hedge": base_metrics[k],
                         "After Hedge": hedged_metrics[k], "Δ": hedged_metrics[k] - base_metrics[k]}
                        for k in base_metrics]
        metric_df = pd.DataFrame(metric_rows).set_index("Metric")
        display_df = metric_df.copy().astype(object)
        for col in ["Before Hedge", "After Hedge", "Δ"]:
            for idx in metric_df.index:
                v = metric_df.loc[idx, col]
                display_df.loc[idx, col] = f"{v:.2f}" if idx == "Sharpe Ratio" else f"{v:.2%}"
        st.dataframe(display_df, use_container_width=True)

    st.markdown("##### Equity Curve & Return Distribution")
    with st.container(border=True):
        c_eq, c_hist = st.columns(2)
        with c_eq:
            base_curve = equity_curve(base_returns)
            hedged_curve = equity_curve(hedged_returns)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=base_curve.index, y=base_curve, name="Before Hedge",
                                      line=dict(color="#2563eb", width=2)))
            fig.add_trace(go.Scatter(x=hedged_curve.index, y=hedged_curve, name="After Hedge",
                                      line=dict(color="#16a34a", width=2)))
            fig.update_layout(title="Equity Curve: Before vs After Hedge", height=380,
                               margin=dict(t=40, b=20), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True, key="hedge_eq")

        with c_hist:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=base_returns, name="Before Hedge", opacity=0.55,
                                        marker_color="#2563eb", nbinsx=60,
                                        histnorm="probability density"))
            fig.add_trace(go.Histogram(x=hedged_returns, name="After Hedge", opacity=0.55,
                                        marker_color="#16a34a", nbinsx=60,
                                        histnorm="probability density"))
            var_before = -historical_var(base_returns, confidence)
            var_after = -historical_var(hedged_returns, confidence)
            fig.add_vline(x=var_before, line_dash="dash", line_color="#2563eb",
                           annotation_text=f"VaR before {var_before:.2%}",
                           annotation_position="top left", annotation_yshift=10)
            fig.add_vline(x=var_after, line_dash="dash", line_color="#16a34a",
                           annotation_text=f"VaR after {var_after:.2%}",
                           annotation_position="top left", annotation_yshift=-15)
            fig.update_layout(title="Daily Return Distribution: Before vs After Hedge",
                               barmode="overlay", height=380, xaxis_tickformat=".1%",
                               margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True, key="hedge_hist")

    st.markdown("##### Drawdown & Tail Risk")
    with st.container(border=True):
        dd_c1, dd_c2 = st.columns(2)
        with dd_c1:
            dd_before = drawdown_series(base_returns)
            dd_after = drawdown_series(hedged_returns)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dd_before.index, y=dd_before, name="Before Hedge",
                                      line=dict(color="#2563eb", width=1)))
            fig.add_trace(go.Scatter(x=dd_after.index, y=dd_after, name="After Hedge",
                                      line=dict(color="#16a34a", width=1)))
            fig.update_layout(title="Drawdown: Before vs After Hedge", height=340,
                               yaxis_tickformat=".0%", margin=dict(t=40, b=20),
                               hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True, key="hedge_dd")

        with dd_c2:
            st.markdown("**Tail Risk Summary**")
            tail_df = pd.DataFrame({
                "Before Hedge": [f"{historical_var(base_returns, confidence):.2%}",
                                  f"{historical_cvar(base_returns, confidence):.2%}",
                                  f"{max_drawdown(base_returns):.2%}"],
                "After Hedge": [f"{historical_var(hedged_returns, confidence):.2%}",
                                 f"{historical_cvar(hedged_returns, confidence):.2%}",
                                 f"{max_drawdown(hedged_returns):.2%}"],
            }, index=[f"VaR {int(confidence*100)}%", f"CVaR {int(confidence*100)}%", "Max Drawdown"])
            st.table(tail_df)
            st.caption("VaR/CVaR are 1-day historical figures on portfolio daily returns. "
                       "A less negative (smaller magnitude) figure after hedging indicates "
                       "reduced tail risk.")


# ------------------------------------------------------------ correlation --
def render_corr_heatmap(corr: pd.DataFrame, title: str, key: str, zrange=(-1, 1)):
    z = corr.values
    fig = go.Figure(data=go.Heatmap(
        z=z, x=list(corr.columns), y=list(corr.index),
        zmin=zrange[0], zmax=zrange[1], colorscale="RdBu", reversescale=True,
        text=np.round(z, 2), texttemplate="%{text}", textfont=dict(size=13),
        colorbar=dict(title="ρ"),
    ))
    fig.update_layout(title=title, height=320 + 26 * len(corr), margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_correlation_tab():
    valid_tickers = [t for t in all_tickers if t not in missing]
    if len(valid_tickers) < 2:
        st.info("Add at least 2 valid tickers (holdings or hedges) in the Controls tab to "
                "compute correlations.")
        return

    R = aligned_returns(prices, valid_tickers)

    st.title("Correlation & Risk Decomposition")
    st.caption(
        f"{len(valid_tickers)} instruments · {R.index.min().date()} → {R.index.max().date()} "
        f"· {len(R)} overlapping trading days (dates on which every instrument has data)"
    )

    sub = st.tabs(["Correlation Matrix", "Rolling Correlation", "Risk Decomposition",
                   "Tail Correlation", "PCA & Clustering"])

    # -- Correlation matrix + significance -----------------------------
    with sub[0]:
        method_label = st.radio(
            "Method", ["Pearson (linear)", "Spearman (rank)", "Kendall (concordance)"],
            horizontal=True, key="corr_method",
        )
        method = {"Pearson (linear)": "pearson", "Spearman (rank)": "spearman",
                  "Kendall (concordance)": "kendall"}[method_label]
        corr = correlation_matrix(R, method=method)

        st.markdown(f"##### {method_label} Correlation Matrix")
        with st.container(border=True):
            render_corr_heatmap(corr, f"{method_label} Correlation", key="corr_heatmap")

        st.markdown("##### Pairwise Statistical Significance (Pearson)")
        with st.container(border=True):
            stats_df = pairwise_stats(R, confidence=confidence)
            ci_label = f"{int(confidence*100)}% CI"
            disp = pd.DataFrame({
                "Pair": stats_df["Pair"],
                "Pearson r": stats_df["Pearson r"].map(lambda v: f"{v:.3f}"),
                ci_label: [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in
                           zip(stats_df["CI Low"], stats_df["CI High"])],
                "p-value": stats_df["p-value"].map(lambda v: f"{v:.2e}" if v >= 1e-300 else "<1e-300"),
                "N": stats_df["N"],
                f"Significant (α={1-confidence:.2f})": stats_df["Significant"].map(
                    lambda b: "Yes" if b else "No"),
            })
            st.dataframe(disp, use_container_width=True, hide_index=True)
            st.caption("Confidence intervals computed via Fisher z-transformation. Null "
                       "hypothesis: ρ = 0. Uses the same confidence level configured in "
                       "Controls → Risk Settings.")

    # -- Rolling correlation --------------------------------------------
    with sub[1]:
        c1, c2, c3 = st.columns([1.5, 1.5, 1])
        asset_a = c1.selectbox("Asset A", valid_tickers, index=0, key="roll_a")
        remaining = [t for t in valid_tickers if t != asset_a] or valid_tickers
        default_b_idx = valid_tickers.index(remaining[0])
        asset_b = c2.selectbox("Asset B", valid_tickers, index=default_b_idx, key="roll_b")
        window = c3.selectbox("Window (days)", [30, 60, 90, 120, 180], index=1, key="roll_window")

        if asset_a == asset_b:
            st.warning("Pick two different instruments to compare.")
        else:
            roll = rolling_correlation(R, asset_a, asset_b, window)
            full_r = float(R[asset_a].corr(R[asset_b]))
            with st.container(border=True):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=roll.index, y=roll, line=dict(color="#3F6C9C", width=2),
                                          name=f"{window}D rolling ρ"))
                fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
                fig.add_hline(y=full_r, line_dash="dash", line_color="#b45309",
                              annotation_text=f"Full-sample ρ = {full_r:.2f}",
                              annotation_position="bottom right")
                fig.update_layout(title=f"{window}-Day Rolling Correlation: {asset_a} vs {asset_b}",
                                   height=400, yaxis_range=[-1, 1], margin=dict(t=40, b=20),
                                   hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True, key="roll_chart")
            st.caption("Correlation is not stationary — a hedge's effectiveness, or a pair's "
                       "diversification benefit, can drift materially over time as the rolling "
                       "window shows.")

    # -- Risk decomposition ----------------------------------------------
    with sub[2]:
        rd_weights = {t: full_weights[t] for t in valid_tickers if t in full_weights}
        if len(rd_weights) < 2:
            st.info("Need at least 2 weighted positions for a risk decomposition.")
        else:
            table, port_vol = risk_decomposition(R, rd_weights)
            dr = diversification_ratio(table, port_vol)

            st.markdown("##### Portfolio Volatility Decomposition")
            with st.container(border=True):
                stat_cols = st.columns(2)
                _metric_cell(stat_cols[0], "Portfolio Ann. Volatility", port_vol, ".2%", NAVY)
                _metric_cell(stat_cols[1], "Diversification Ratio", dr, ".2f", NAVY)
                st.caption("Diversification Ratio = (Σ |wᵢ|·σᵢ) / σ_portfolio — the weighted "
                           "average of standalone volatilities divided by actual portfolio "
                           "volatility. Values above 1 mean correlation/hedging is reducing risk "
                           "below the naive sum of the parts; 1.0 means no diversification benefit.")

                disp = pd.DataFrame({
                    "Weight": table["Weight"].map(lambda v: f"{v:.1%}"),
                    "Standalone Ann. Vol": table["Standalone Ann. Vol"].map(lambda v: f"{v:.1%}"),
                    "MCTR": table["MCTR"].map(lambda v: f"{v:.4f}"),
                    "CCTR (Ann. Vol)": table["CCTR (Ann. Vol)"].map(lambda v: f"{v:.2%}"),
                    "% of Portfolio Risk": table["% of Portfolio Risk"].map(lambda v: f"{v:.1f}%"),
                }, index=table.index)
                st.dataframe(disp, use_container_width=True)
                st.caption("MCTR = ∂σₚ/∂wᵢ (marginal contribution to risk). CCTR = wᵢ · MCTRᵢ; "
                           "component contributions sum exactly to portfolio volatility (Euler's "
                           "homogeneous-function decomposition — verify: the % column sums to "
                           "100%). A negative %  means that position is currently reducing total "
                           "portfolio risk, i.e. it's acting as a genuine hedge right now.")

            st.markdown("##### % Contribution to Portfolio Risk")
            with st.container(border=True):
                colors = ["#dc2626" if v < 0 else "#3F6C9C" for v in table["% of Portfolio Risk"]]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(table.index), y=table["% of Portfolio Risk"], marker_color=colors,
                    text=[f"{v:.1f}%" for v in table["% of Portfolio Risk"]], textposition="outside",
                ))
                fig.add_hline(y=0, line_color="#94a3b8")
                fig.update_layout(height=360, margin=dict(t=30, b=20),
                                   yaxis_title="% of Portfolio Risk")
                st.plotly_chart(fig, use_container_width=True, key="risk_contrib_chart")

    # -- Tail / conditional correlation ----------------------------------
    with sub[3]:
        quantile = st.slider("Stress quantile (worst X% of portfolio days)", 0.05, 0.25, 0.10,
                              step=0.01, key="tail_quantile")
        ref_weights = {t: full_weights[t] for t in valid_tickers if t in full_weights}
        port_returns_full = portfolio_daily_returns(prices[list(ref_weights)], ref_weights)
        full_corr, stress_corr, n_days, thresh = conditional_correlation(
            R, port_returns_full, quantile=quantile)

        st.caption(f"Stress sample: {n_days} days where the portfolio return was ≤ {thresh:.2%} "
                   f"(the worst {quantile:.0%} of days in the overlapping sample).")

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                render_corr_heatmap(full_corr, "Full-Sample Correlation", key="full_corr_hm")
        with c2:
            with st.container(border=True):
                render_corr_heatmap(stress_corr, "Stress-Day Correlation", key="stress_corr_hm")

        st.markdown("##### Correlation Shift in Stress (Stress − Full-Sample)")
        with st.container(border=True):
            delta = stress_corr - full_corr
            max_abs = float(np.nanmax(np.abs(delta.values))) or 1.0
            render_corr_heatmap(delta, "Δ Correlation (Stress − Full)", key="delta_corr_hm",
                                 zrange=(-max_abs, max_abs))
            st.caption("Positive values (red/orange) mean two assets become MORE correlated during the "
                       "portfolio's worst days than on average — the classic 'correlations go to 1 "
                       "in a crisis' effect that erodes diversification exactly when it's needed "
                       "most. This is the quantitative version of the downturn risk discussed on "
                       "the Portfolio tab's hedge comparison.")

    # -- PCA & hierarchical clustering ------------------------------------
    with sub[4]:
        corr_pearson = correlation_matrix(R, method="pearson")
        explained, loadings, eigenvalues = pca_decomposition(corr_pearson)

        st.markdown("##### Explained Variance (Scree Plot)")
        with st.container(border=True):
            labels = [f"PC{i+1}" for i in range(len(explained))]
            cum = np.cumsum(explained) * 100
            fig = go.Figure()
            fig.add_trace(go.Bar(x=labels, y=explained * 100, marker_color="#3F6C9C",
                                  name="Individual",
                                  text=[f"{v:.0%}" for v in explained], textposition="outside"))
            fig.add_trace(go.Scatter(x=labels, y=cum, mode="lines+markers", name="Cumulative",
                                      line=dict(color="#b45309", dash="dot")))
            fig.update_layout(height=360, margin=dict(t=30, b=20),
                               yaxis_title="% Variance Explained", legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True, key="scree_chart")
            st.caption(
                f"PC1 alone explains {explained[0]:.0%} of the correlation structure across your "
                f"{len(valid_tickers)} instruments — a rough proxy for how much of the book's risk "
                "comes from one dominant common factor versus genuinely independent bets. "
                "Eigenvalues: " + ", ".join(f"{v:.2f}" for v in eigenvalues) +
                f" (sum = {eigenvalues.sum():.1f} = number of instruments, as expected for a "
                "correlation matrix)."
            )

        st.markdown("##### Factor Loadings")
        with st.container(border=True):
            n_show = min(3, loadings.shape[1])
            render_corr_heatmap(loadings.iloc[:, :n_show], "PC Loadings", key="loadings_hm")
            st.caption("Assets with the same sign on a component move together on that factor; "
                       "opposite signs move against each other on that factor.")

        st.markdown("##### Hierarchical Clustering (Correlation Distance)")
        with st.container(border=True):
            dendro = dendrogram_figure(corr_pearson)
            tickvals = list(dendro.layout.xaxis.tickvals or [])
            if len(tickvals) > 1:
                pad = (tickvals[1] - tickvals[0]) / 2
                dendro.update_layout(xaxis_range=[min(tickvals) - pad, max(tickvals) + pad])
            dendro.update_layout(height=380, margin=dict(t=30, b=20), showlegend=False)
            st.plotly_chart(dendro, use_container_width=True, key="dendrogram_chart")
            st.caption("Distance = √(0.5·(1−ρ)), average linkage. Assets that merge at low height "
                       "are highly correlated and offer little diversification against each other; "
                       "assets that merge only at the top are close to independent.")


# --------------------------------------------------------- risk scenarios --
SCENARIO_DEFS = [("Down 10%", -0.10), ("Down 5%", -0.05), ("Down 1%", -0.01),
                  ("Up 1%", 0.01), ("Up 5%", 0.05), ("Up 10%", 0.10)]


def render_scenario_content(pct: float, mm_base: dict, mm_full: dict, base_port_ret: pd.Series,
                             full_port_ret: pd.Series, has_hedge: bool, ind_betas: dict):
    st.markdown(f"##### Scenario: Economy (SPY) moves {pct:+.0%}")

    rows = []
    for name, mm, ret in [("Before Hedge", mm_base, base_port_ret)] + (
            [("After Hedge", mm_full, full_port_ret)] if has_hedge else []):
        svar, scvar, mu = stressed_var_cvar(mm["beta"], pct, mm["residual_std"], confidence)
        rows.append({
            "Portfolio": name, "Beta to SPY": mm["beta"], "Shock P&L": mu,
            "Baseline VaR": historical_var(ret, confidence),
            "Stressed VaR": svar,
            "Baseline CVaR": historical_cvar(ret, confidence),
            "Stressed CVaR": scvar,
        })
    df = pd.DataFrame(rows).set_index("Portfolio")

    with st.container(border=True):
        cols = st.columns(len(rows))
        for col, (name, r) in zip(cols, zip(df.index, rows)):
            _metric_cell(col, f"{name}: Scenario Shock P&L", r["Shock P&L"], "+.2%",
                         GREEN if r["Shock P&L"] >= 0 else RED)

        cl = f"{int(confidence*100)}%"
        disp = pd.DataFrame({
            "Beta to SPY": df["Beta to SPY"].map(lambda v: f"{v:.3f}"),
            "Shock P&L": df["Shock P&L"].map(lambda v: f"{v:+.2%}"),
            f"Baseline VaR {cl}": df["Baseline VaR"].map(lambda v: f"{v:.2%}"),
            f"Stressed VaR {cl}": df["Stressed VaR"].map(
                lambda v: f"{v:.2%}" if v >= 0 else f"({-v:.2%} gain)"),
            f"Baseline CVaR {cl}": df["Baseline CVaR"].map(lambda v: f"{v:.2%}"),
            f"Stressed CVaR {cl}": df["Stressed CVaR"].map(
                lambda v: f"{v:.2%}" if v >= 0 else f"({-v:.2%} gain)"),
        }, index=df.index)
        st.dataframe(disp, use_container_width=True)
        st.caption(
            "Shock P&L = βₚ × scenario move, from a single-factor OLS regression of portfolio "
            "returns on SPY. Stressed VaR/CVaR re-center the portfolio's residual (idiosyncratic, "
            "market-orthogonal) volatility around that shock: SVaR = -(μ+z·σ), "
            "SCVaR = -(μ-σ·φ(z)/(1-c)), where μ = shock P&L, σ = daily residual std, "
            "z = Φ⁻¹(1-c). Both formulas verified against 2M-draw Monte Carlo simulation. A "
            "negative value means even the worst-case tail under this scenario is still a gain."
        )

    st.markdown("###### Per-Instrument Shock Contribution (Full Portfolio, incl. hedge)")
    with st.container(border=True):
        rows2 = []
        for t, w in full_weights.items():
            b = ind_betas.get(t, np.nan)
            rows2.append({"Ticker": t, "Weight": w, "Beta to SPY": b, "Contribution": w * b * pct})
        df2 = pd.DataFrame(rows2).set_index("Ticker")
        colors = ["#dc2626" if v < 0 else "#3F6C9C" for v in df2["Contribution"]]
        fig = go.Figure(go.Bar(x=list(df2.index), y=df2["Contribution"] * 100, marker_color=colors,
                                text=[f"{v:+.2%}" for v in df2["Contribution"]], textposition="outside"))
        fig.add_hline(y=0, line_color="#94a3b8")
        fig.update_layout(height=320, margin=dict(t=20, b=20), yaxis_title="Contribution to Shock P&L (%)")
        st.plotly_chart(fig, use_container_width=True, key=f"shock_contrib_{pct}")
        disp2 = pd.DataFrame({
            "Weight": df2["Weight"].map(lambda v: f"{v:+.1%}"),
            "Beta to SPY": df2["Beta to SPY"].map(lambda v: f"{v:.3f}"),
            "Contribution to Shock P&L": df2["Contribution"].map(lambda v: f"{v:+.3%}"),
        }, index=df2.index)
        st.dataframe(disp2, use_container_width=True)
        st.caption(f"Contributions sum to {df2['Contribution'].sum():+.3%}, matching the full "
                   f"portfolio's Shock P&L above exactly (weighted-beta decomposition).")


def render_optimal_hedge_content(base_port_ret: pd.Series, merged_rets: pd.DataFrame,
                                  candidate_tickers: list):
    st.markdown("##### Optimal (Minimum-Variance) Hedge Ratio")
    default_candidate = next(iter(hedge_weights_input), candidate_tickers[0])
    default_idx = candidate_tickers.index(default_candidate) if default_candidate in candidate_tickers else 0
    candidate = st.selectbox("Hedge candidate instrument", candidate_tickers, index=default_idx,
                              key="opt_hedge_candidate")
    hedge_ret = merged_rets[candidate]
    result = optimal_hedge_ratio(base_port_ret, hedge_ret)
    current_h = hedge_weights_input.get(candidate, 0.0)
    current_var = hedged_variance(base_port_ret, hedge_ret, current_h)
    current_vol = float(np.sqrt(current_var * 252))
    reduction_current = 1 - current_var / result["var_unhedged"] if result["var_unhedged"] else np.nan
    reduction_optimal = result["hedge_effectiveness"]

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        _metric_cell(c1, "Optimal Hedge Weight (h*)", result["h_star"], "+.1%", NAVY)
        _metric_cell(c2, "Current Configured Weight", current_h, "+.1%", NAVY)
        _metric_cell(c3, "Correlation (ρ)", result["rho"], "+.3f", NAVY)
        _metric_cell(c4, "Max Hedge Effectiveness (ρ²)", result["hedge_effectiveness"], ".1%", NAVY)
        st.caption("h* = -Cov(portfolio, hedge) / Var(hedge) (Ederington 1979) — the weight that "
                   "minimizes portfolio variance. Verified against brute-force grid search over "
                   "20,000 candidate weights. Hedge Effectiveness = ρ² = the maximum fraction of "
                   "portfolio variance this single instrument can remove.")

    with st.container(border=True):
        hs, variances = variance_curve(base_port_ret, hedge_ret)
        vols = np.sqrt(variances * 252) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hs * 100, y=vols, mode="lines",
                                  line=dict(color="#3F6C9C", width=2.5), name="Portfolio Vol"))
        fig.add_vline(x=result["h_star"] * 100, line_dash="dash", line_color="#16a34a",
                      annotation_text=f"Optimal h* = {result['h_star']:+.1%}",
                      annotation_position="top")
        fig.add_vline(x=current_h * 100, line_dash="dot", line_color="#b45309",
                      annotation_text=f"Current h = {current_h:+.1%}",
                      annotation_position="bottom")
        fig.update_layout(title=f"Portfolio Volatility vs. {candidate} Hedge Weight",
                           xaxis_title="Hedge Weight (%)", yaxis_title="Annualized Portfolio Vol (%)",
                           height=420, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True, key="hedge_variance_curve")

    with st.container(border=True):
        summary = pd.DataFrame({
            "Ann. Volatility": [f"{result['vol_unhedged']:.2%}", f"{current_vol:.2%}",
                                 f"{result['vol_at_optimal']:.2%}"],
        }, index=["Unhedged", f"Current ({current_h:+.1%})", f"Optimal ({result['h_star']:+.1%})"])
        st.dataframe(summary, use_container_width=True)
        pct_of_max = reduction_current / reduction_optimal if reduction_optimal else np.nan
        st.caption(f"The current hedge weight captures {pct_of_max:.0%} of the maximum achievable "
                   f"variance reduction from {candidate} ({reduction_current:.1%} realized vs. a "
                   f"maximum possible {reduction_optimal:.1%} at the optimal weight).")


def render_benchmark_content(base_port_ret: pd.Series, full_port_ret: pd.Series,
                              spy_ret: pd.Series, ief_ret: pd.Series, has_hedge: bool):
    st.markdown("##### Benchmark Comparison")
    c1, c2 = st.columns(2)
    bench_choice = c1.radio("Benchmark", ["SPY (S&P 500)", "60/40 SPY/IEF Blend"],
                             horizontal=True, key="bench_choice")
    port_choice = c2.radio("Portfolio", ["After Hedge", "Before Hedge"] if has_hedge else ["Portfolio"],
                            horizontal=True, key="bench_port_choice")
    bench_ret = (0.6 * spy_ret + 0.4 * ief_ret) if bench_choice.startswith("60/40") else spy_ret
    bench_label = "60/40 SPY/IEF" if bench_choice.startswith("60/40") else "SPY"
    port_ret = base_port_ret if port_choice == "Before Hedge" else full_port_ret

    stats = benchmark_stats(port_ret, bench_ret)
    with st.container(border=True):
        cols = st.columns(4)
        _metric_cell(cols[0], "Beta", stats["beta"], ".3f", NAVY)
        _metric_cell(cols[1], "Annualized Alpha", stats["alpha_annual"], "+.2%",
                      GREEN if stats["alpha_annual"] >= 0 else RED)
        _metric_cell(cols[2], "R²", stats["r_squared"], ".1%", NAVY)
        _metric_cell(cols[3], "Tracking Error", stats["tracking_error"], ".2%", AMBER)
        cols2 = st.columns(4)
        _metric_cell(cols2[0], "Information Ratio", stats["information_ratio"], "+.2f",
                      GREEN if stats["information_ratio"] >= 0 else RED)
        _metric_cell(cols2[1], "Up-Market Capture", stats["up_capture"], ".1%", NAVY)
        _metric_cell(cols2[2], "Down-Market Capture", stats["down_capture"], ".1%", NAVY)
        _metric_cell(cols2[3], "N (days)", stats["n"], ".0f", NAVY)
        st.caption(f"{port_choice} vs. {bench_label}. Alpha is annualized CAPM alpha (arithmetic "
                   "scaling of the daily OLS intercept). Up/Down capture = average portfolio return "
                   "on days the benchmark rose/fell, divided by the benchmark's average return on "
                   "those same days.")

    with st.container(border=True):
        aligned = pd.concat([port_ret.rename("p"), bench_ret.rename("b")], axis=1).dropna()
        port_curve = (1 + aligned["p"]).cumprod() * 100
        bench_curve = (1 + aligned["b"]).cumprod() * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port_curve.index, y=port_curve, name=port_choice,
                                  line=dict(color="#3F6C9C", width=2.5)))
        fig.add_trace(go.Scatter(x=bench_curve.index, y=bench_curve, name=bench_label,
                                  line=dict(color="#b45309", width=2)))
        fig.update_layout(title=f"{port_choice} vs. {bench_label} (base = 100)", height=400,
                           margin=dict(t=40, b=20), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, key="benchmark_curve")


def _apply_config(loaded: dict):
    next_id = st.session_state.next_id
    new_holdings = []
    for i, h in enumerate(loaded.get("holdings", [])):
        next_id += 1
        new_holdings.append({"id": f"h{next_id}", "ticker": str(h["ticker"]).strip().upper(),
                              "default_weight": float(h["weight"]), "default_dir": h["direction"]})
    new_hedges = []
    for i, g in enumerate(loaded.get("hedges", [])):
        next_id += 1
        new_hedges.append({"id": f"g{next_id}", "ticker": str(g["ticker"]).strip().upper(),
                            "default_weight": float(g["weight"]), "default_dir": g["direction"]})

    for key in list(st.session_state.keys()):
        if key.startswith(("w_", "wn_", "d_", "rm_")):
            del st.session_state[key]

    st.session_state.holdings_meta = new_holdings
    st.session_state.hedges_meta = new_hedges
    st.session_state.next_id = next_id
    st.session_state["risk_free"] = float(loaded.get("risk_free_pct", 4.0))
    st.session_state["confidence_pct"] = int(loaded.get("confidence_pct", 95))
    st.session_state["_config_loaded"] = True


def render_save_export_content():
    st.markdown("##### Save Configuration")
    with st.container(border=True):
        config = {
            "holdings": [{"ticker": h["ticker"],
                          "weight": st.session_state.get(f"w_{h['id']}", h["default_weight"]),
                          "direction": st.session_state.get(f"d_{h['id']}", h["default_dir"])}
                         for h in st.session_state.holdings_meta],
            "hedges": [{"ticker": g["ticker"],
                        "weight": st.session_state.get(f"w_{g['id']}", g["default_weight"]),
                        "direction": st.session_state.get(f"d_{g['id']}", g["default_dir"])}
                       for g in st.session_state.hedges_meta],
            "risk_free_pct": st.session_state.get("risk_free", 4.0),
            "confidence_pct": st.session_state.get("confidence_pct", 95),
        }
        json_str = json.dumps(config, indent=2)
        st.download_button("Download configuration (JSON)", data=json_str,
                            file_name="portfolio_config.json", mime="application/json",
                            key="download_config_btn")
        st.code(json_str, language="json")

    st.markdown("##### Load Configuration")
    with st.container(border=True):
        if st.session_state.pop("_config_loaded", False):
            st.success("Configuration loaded — check the Controls tab to review, or any other tab "
                       "to see it reflected live.")
        uploaded = st.file_uploader("Upload a portfolio_config.json file", type=["json"],
                                     key="config_upload")
        if uploaded is not None:
            try:
                loaded = json.load(uploaded)
                st.button("Apply this configuration", key="apply_config_btn",
                          on_click=_apply_config, args=(loaded,))
            except Exception as e:
                st.error(f"Could not parse file: {e}")


def render_risk_scenarios_tab():
    valid_tickers = [t for t in all_tickers if t not in missing]
    if not base_weights or len(valid_tickers) < 1:
        st.warning("Add at least one valid holding in the Controls tab to run risk scenarios.")
        return

    with st.spinner("Pulling benchmark data (SPY, IEF) from Yahoo Finance..."):
        try:
            bench_prices = fetch_prices(("SPY", "IEF"), period="max")
        except Exception as e:
            st.error(f"Failed to fetch benchmark data: {e}")
            return

    needed = sorted(set(base_weights) | set(full_weights))
    merged_prices = prices[needed].join(bench_prices, how="inner")
    merged_rets = merged_prices.pct_change().dropna(how="any")

    base_port_ret = (merged_rets[list(base_weights)] * pd.Series(base_weights)).sum(axis=1)
    full_port_ret = (merged_rets[list(full_weights)] * pd.Series(full_weights)).sum(axis=1)
    spy_ret, ief_ret = merged_rets["SPY"], merged_rets["IEF"]
    has_hedge = bool(hedge_weights_input)

    mm_base = market_model(base_port_ret, spy_ret)
    mm_full = market_model(full_port_ret, spy_ret) if has_hedge else mm_base
    ind_betas = asset_betas(merged_rets, spy_ret, list(full_weights))

    st.title("Risk Scenarios: VaR & Stressed VaR")
    st.caption(
        f"Single-factor market model vs. SPY · {mm_full['n']} overlapping trading days · "
        f"Full portfolio β = {mm_full['beta']:.3f} (R² = {mm_full['r_squared']:.1%}) · "
        f"Confidence level {int(confidence*100)}% (set in Controls → Risk Settings)."
    )

    sub_labels = [s[0] for s in SCENARIO_DEFS] + ["Optimal Hedge", "Benchmark Comparison", "Save / Export"]
    sub = st.tabs(sub_labels)

    for (_, pct), tab in zip(SCENARIO_DEFS, sub[:6]):
        with tab:
            render_scenario_content(pct, mm_base, mm_full, base_port_ret, full_port_ret,
                                     has_hedge, ind_betas)

    with sub[6]:
        render_optimal_hedge_content(base_port_ret, merged_rets, valid_tickers)

    with sub[7]:
        render_benchmark_content(base_port_ret, full_port_ret, spy_ret, ief_ret, has_hedge)

    with sub[8]:
        render_save_export_content()


def _sync_weight(item: dict, source: str):
    w_key, n_key = f"w_{item['id']}", f"wn_{item['id']}"
    if source == "slider":
        st.session_state[n_key] = st.session_state[w_key]
    else:
        st.session_state[w_key] = st.session_state[n_key]


def render_position_row(item: dict, group_key: str):
    w_key, n_key, d_key = f"w_{item['id']}", f"wn_{item['id']}", f"d_{item['id']}"
    # Seed defaults once, then never pass value=/default= alongside a key that
    # a callback (sync or config-load) might also write to — Streamlit warns
    # (and can misbehave) if both happen in the same run.
    st.session_state.setdefault(w_key, item["default_weight"])
    st.session_state.setdefault(n_key, item["default_weight"])
    st.session_state.setdefault(d_key, item["default_dir"])

    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns([1.3, 2.0, 1.0, 2.0, 0.5])
        c1.markdown(f"**{item['ticker']}**")
        c2.slider("Weight %", 0.0, 100.0, step=0.5, key=w_key, label_visibility="collapsed",
                  on_change=_sync_weight, args=(item, "slider"))
        c3.number_input("Weight % (exact)", 0.0, 100.0, step=0.5, key=n_key,
                         label_visibility="collapsed", on_change=_sync_weight, args=(item, "number"))
        c4.segmented_control("Direction", ["Long", "Short"], key=d_key, required=True,
                              label_visibility="collapsed")
        remove = c5.button("✕", key=f"rm_{item['id']}", help="Remove")
    return remove


def render_add_form(group_key: str, meta_key: str, default_dir: str):
    with st.expander(f"Add {'holding' if group_key == 'holding' else 'hedge instrument'}"):
        c1, c2, c3 = st.columns([2, 1.5, 1.5])
        ticker = c1.text_input("Ticker", key=f"new_{group_key}_ticker", autocomplete="off")
        weight = c2.number_input("Weight %", min_value=0.0, max_value=100.0, value=10.0,
                                  step=0.5, key=f"new_{group_key}_weight")
        direction = c3.segmented_control("Direction", ["Long", "Short"], default=default_dir,
                                          key=f"new_{group_key}_dir", required=True)
        if st.button(f"Add {group_key}", key=f"add_{group_key}_btn"):
            if ticker.strip():
                st.session_state.next_id += 1
                st.session_state[meta_key].append({
                    "id": f"{group_key[0]}{st.session_state.next_id}",
                    "ticker": ticker.strip().upper(),
                    "default_weight": weight,
                    "default_dir": direction or default_dir,
                })
                st.rerun()
            else:
                st.warning("Enter a ticker symbol first.")


def render_controls_tab():
    st.title("Controls")
    st.caption("Adjust position sizes, direction, and risk settings — every other "
               "tab updates live.")

    st.markdown("#### Holdings")
    remove_id = None
    for item in st.session_state.holdings_meta:
        if render_position_row(item, "holding"):
            remove_id = item["id"]
    if remove_id:
        st.session_state.holdings_meta = [h for h in st.session_state.holdings_meta
                                           if h["id"] != remove_id]
        st.rerun()
    render_add_form("holding", "holdings_meta", "Long")

    st.divider()
    st.markdown("#### Hedge Instruments")
    remove_id = None
    for item in st.session_state.hedges_meta:
        if render_position_row(item, "hedge"):
            remove_id = item["id"]
    if remove_id:
        st.session_state.hedges_meta = [g for g in st.session_state.hedges_meta
                                         if g["id"] != remove_id]
        st.rerun()
    render_add_form("hedge", "hedges_meta", "Short")

    st.divider()
    st.markdown("#### Risk Settings")
    with st.container(border=True):
        st.session_state.setdefault("risk_free", 4.0)
        st.session_state.setdefault("confidence_pct", 95)
        c1, c2 = st.columns(2)
        c1.slider("Risk-free rate (annual %)", 0.0, 10.0, step=0.25, key="risk_free")
        c2.slider("VaR / CVaR confidence (%)", 90, 99, step=1, key="confidence_pct")

    st.divider()
    st.markdown("#### Live Exposure Summary")
    with st.container(border=True):
        b_net, b_gross = net_exposure(base_weights_raw), gross_exposure(base_weights_raw)
        h_net, h_gross = net_exposure(hedge_weights_raw), gross_exposure(hedge_weights_raw)
        c1, c2 = st.columns(2)
        c1.metric("Base Net / Gross Exposure", f"{b_net:.0%} / {b_gross:.0%}")
        c2.metric("Hedge Net / Gross Exposure", f"{h_net:.0%} / {h_gross:.0%}")


# ------------------------------------------------------------------- tabs --
tab_tickers = list(dict.fromkeys(
    [h["ticker"] for h in holdings] + [g["ticker"] for g in hedges]
))
tab_labels = ["Portfolio"] + tab_tickers + ["Correlation", "Risk Scenarios", "Controls"]
tabs = st.tabs(tab_labels)

with tabs[0]:
    render_portfolio_tab()

for ticker, tab in zip(tab_tickers, tabs[1:len(tab_tickers) + 1]):
    with tab:
        render_ticker_tab(ticker)

with tabs[-3]:
    render_correlation_tab()

with tabs[-2]:
    render_risk_scenarios_tab()

with tabs[-1]:
    render_controls_tab()
