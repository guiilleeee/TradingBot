"""
account_manager.py — Fetches live account equity from brokers (Alpaca, OKX).

Used to replace static config values with real funded capital before a run.
Falls back to the config value if API keys are missing or requests fail.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import ccxt
import requests

logger = logging.getLogger(__name__)

ALPACA_URL = "https://api.alpaca.markets/v2/account"
# If using paper trading, you would use "https://paper-api.alpaca.markets/v2/account"


def get_alpaca_equity() -> Optional[float]:
    """Fetch equity from Alpaca (equity/stocks broker)."""
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")

    if not api_key or not api_secret:
        logger.debug("Alpaca keys missing, skipping Alpaca equity fetch.")
        return None

    try:
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }
        # Defaulting to live Alpaca URL, though we might use paper for testing.
        resp = requests.get(ALPACA_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        
        data = resp.json()
        equity = float(data.get("equity", 0.0))
        logger.info("Fetched Alpaca live equity: $%.2f", equity)
        return equity
    except Exception as exc:
        logger.warning("Failed to fetch Alpaca equity: %s", exc)
        return None


def get_okx_equity() -> Optional[float]:
    """Fetch total equity from OKX (crypto broker) via CCXT."""
    api_key = os.getenv("OKX_API_KEY")
    secret_key = os.getenv("OKX_SECRET_KEY")
    passphrase = os.getenv("OKX_PASSPHRASE")

    if not api_key or not secret_key or not passphrase:
        logger.debug("OKX keys missing, skipping OKX equity fetch.")
        return None

    try:
        exchange = ccxt.okx({
            "apiKey": api_key,
            "secret": secret_key,
            "password": passphrase,
            "enableRateLimit": True,
        })
        balance = exchange.fetch_balance()
        
        # 'info' holds the raw response, which for OKX v5 contains 'totalEq'
        total_eq = 0.0
        info = balance.get("info")
        if isinstance(info, dict):
            data = info.get("data")
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                total_eq_str = data[0].get("totalEq", "0.0")
                total_eq = float(total_eq_str)
            else:
                logger.warning("Could not parse OKX balance data list structure. Defaulting to 0.0.")
        else:
            logger.warning("Could not parse OKX balance info structure. Defaulting to 0.0.")

        logger.info("Fetched OKX live equity: $%.2f", total_eq)
        return total_eq
    except Exception as exc:
        logger.warning("Failed to fetch OKX equity: %s", exc)
        return None


def fetch_live_equity(fallback_equity: float) -> float:
    """
    Fetch and sum real equity from all brokers.
    Falls back to `fallback_equity` if no brokers succeed or are configured.
    """
    alpaca_eq = get_alpaca_equity()
    okx_eq = get_okx_equity()
    
    total = 0.0
    success = False
    
    if alpaca_eq is not None:
        total += alpaca_eq
        success = True
        
    if okx_eq is not None:
        total += okx_eq
        success = True
        
    if success:
        logger.info("Total live account equity across brokers: $%.2f", total)
        return total
    else:
        logger.warning("Could not fetch live equity from any broker. Using fallback: $%.2f", fallback_equity)
        return fallback_equity
