"""
data_fetcher.py — Pulls OHLCV data from yfinance and headlines from
Yahoo Finance RSS, then assembles a validated SignalInput payload.

No API key required for either source.
"""
from __future__ import annotations

import logging
from typing import Optional

import feedparser
import pandas as pd
import yfinance as yf

from .models import ExistingPosition, SignalInput, TechnicalIndicators

logger = logging.getLogger(__name__)


# ─── OHLCV + Indicators ───────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, period: str = "60d", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV history via yfinance. Returns a DataFrame indexed by date."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)
    df = df.dropna(subset=['Close'])
    if df.empty:
        raise ValueError(f"No OHLCV data returned for symbol '{symbol}'. "
                         "Check that the symbol is correct and markets are not closed.")
    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Wilder's RSI — no TA-Lib dependency."""
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Initial averages (simple mean for the seed)
    avg_gain = gain.head(period).mean()
    avg_loss = loss.head(period).mean()

    # Wilder smoothing over remaining bars
    for i in range(period, len(gain)):
        avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def compute_indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    sma_short: int = 20,
    sma_long: int = 50,
) -> TechnicalIndicators:
    """Derive all technical indicators from an OHLCV DataFrame."""
    close = df["Close"]
    volume = df["Volume"]

    if len(close) < sma_long + 1:
        raise ValueError(
            f"Not enough history to compute SMA-{sma_long}. "
            f"Got {len(close)} bars; need at least {sma_long + 1}."
        )

    rsi = _compute_rsi(close, rsi_period)
    sma_20_val = float(close.rolling(sma_short).mean().iloc[-1])
    sma_50_val = float(close.rolling(sma_long).mean().iloc[-1])

    current_close = float(close.iloc[-1])
    prev_close    = float(close.iloc[-2])
    price_change_24h_pct = round((current_close - prev_close) / prev_close * 100, 4)

    current_vol = float(volume.iloc[-1])
    prev_vol    = float(volume.iloc[-2])
    volume_change_24h_pct = (
        round((current_vol - prev_vol) / prev_vol * 100, 4) if prev_vol != 0 else 0.0
    )

    return TechnicalIndicators(
        rsi_14=rsi,
        sma_20=round(sma_20_val, 4),
        sma_50=round(sma_50_val, 4),
        price_change_24h_pct=price_change_24h_pct,
        volume_change_24h_pct=volume_change_24h_pct,
    )


def get_current_price(df: pd.DataFrame) -> float:
    """Return the most recent closing price."""
    return float(df["Close"].iloc[-1])


# ─── Headlines ────────────────────────────────────────────────────────────────

_YAHOO_RSS_TEMPLATE = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"


def fetch_headlines(symbol: str, max_headlines: int = 6) -> list[str]:
    """
    Pull recent headlines from Yahoo Finance RSS.
    Returns an empty list gracefully if the feed is unavailable.
    """
    # Yahoo Finance RSS uses the raw ticker without the exchange suffix for crypto
    clean_symbol = symbol.replace("-USD", "").replace("-USDT", "")
    url = _YAHOO_RSS_TEMPLATE.format(symbol=clean_symbol)

    try:
        feed = feedparser.parse(url)
        titles = [str(entry.get("title", "")) for entry in feed.entries[:max_headlines]]
        titles = [t for t in titles if t]
        logger.debug("Fetched %d headlines for %s", len(titles), symbol)
        return titles
    except Exception as exc:
        logger.warning("Could not fetch headlines for %s: %s", symbol, exc)
        return []


# ─── Main builder ─────────────────────────────────────────────────────────────

def build_signal_input(
    symbol: str,
    asset_class: str,
    account_equity_usd: float,
    existing_position: Optional[ExistingPosition] = None,
    ohlcv_period: str = "60d",
    rsi_period: int = 14,
    sma_short: int = 20,
    sma_long: int = 50,
    max_headlines: int = 6,
) -> SignalInput:
    """
    High-level helper: fetch data, compute indicators, gather headlines,
    and return a fully-validated SignalInput ready to send to Claude.
    """
    logger.info("Fetching market data for %s …", symbol)
    df = fetch_ohlcv(symbol, period=ohlcv_period)

    current_price = get_current_price(df)
    indicators = compute_indicators(df, rsi_period=rsi_period, sma_short=sma_short, sma_long=sma_long)
    headlines = fetch_headlines(symbol, max_headlines=max_headlines)

    return SignalInput(
        symbol=symbol,
        asset_class=asset_class,
        current_price=round(current_price, 4),
        account_equity_usd=account_equity_usd,
        existing_position=existing_position,
        technical_indicators=indicators,
        recent_headlines=headlines,
    )
