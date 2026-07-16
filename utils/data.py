"""Yahoo Finance data access via yfinance."""
import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers: tuple, period: str = "max") -> pd.DataFrame:
    """Adjusted close prices for one or more tickers, aligned and forward-filled."""
    raw = yf.download(list(tickers), period=period, auto_adjust=True,
                       progress=False, group_by="ticker")

    if len(tickers) == 1:
        close = raw["Close"].to_frame(name=tickers[0])
    else:
        close = pd.DataFrame({t: raw[t]["Close"] for t in tickers})

    close = close.sort_index().ffill().dropna(how="all")
    return close


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news(ticker: str, limit: int = 6) -> list:
    """Recent headlines for a ticker. Futures/index symbols often have none."""
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []

    items = []
    for entry in raw[:limit]:
        content = entry.get("content", {})
        url = ((content.get("canonicalUrl") or {}).get("url")
               or (content.get("clickThroughUrl") or {}).get("url"))
        items.append({
            "title": content.get("title", ""),
            "publisher": (content.get("provider") or {}).get("displayName", ""),
            "published": content.get("pubDate", ""),
            "url": url,
        })
    return items


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_display_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).get_info()
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker


@st.cache_data(ttl=300, show_spinner=False)
def fetch_quote(ticker: str) -> dict:
    try:
        fi = yf.Ticker(ticker).fast_info
        last = fi.get("lastPrice")
        prev = fi.get("previousClose")
        change_pct = (last - prev) / prev if last is not None and prev else None
        return {"last": last, "prev_close": prev, "change_pct": change_pct,
                "currency": fi.get("currency", "")}
    except Exception:
        return {"last": None, "prev_close": None, "change_pct": None, "currency": ""}
