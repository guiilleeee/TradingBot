"""
run_gemini_test.py — Isolated execution loop for the Gemini 7-day test.
Never makes real trades, logs to a separate database.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any
from datetime import datetime, timezone, timedelta

import yaml
from dotenv import load_dotenv

from .account_manager import fetch_live_equity
from .data_fetcher import build_signal_input
from .logger import BotLogger
from .risk_manager import validate
from .signal_generator_gemini import generate_signal
from .models import ExecutionResult

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("src.run_gemini_test")


def load_config(path: Path = Path("config.yaml")) -> dict[str, Any]:
    if not path.exists():
        logger.error("Config file not found at %s", path)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_self_termination(bot_logger: BotLogger) -> bool:
    """Returns True if the test should stop (more than 7 days since first run)."""
    with bot_logger._connect() as conn:
        cursor = conn.execute("SELECT MIN(timestamp) as first_ts FROM signals")
        row = cursor.fetchone()
        if not row or not row["first_ts"]:
            return False  # No runs yet

        try:
            first_ts = datetime.fromisoformat(row["first_ts"])
        except ValueError:
            return False

        if datetime.now(timezone.utc) - first_ts > timedelta(days=7):
            return True
    return False


def process_symbol(symbol: str, asset_class: str, config: dict[str, Any], equity: float, bot_logger: BotLogger) -> None:
    """End-to-end flow for a single symbol for the Gemini test."""
    cb_threshold = config["circuit_breaker_loss_pct"]
    max_risk = config.get("max_risk_pct", 2.0)
    max_abs_pos = config.get("max_absolute_position_pct", 30.0)
    min_conf = config.get("min_confidence", 0.55)

    try:
        # In this isolated test, we do NOT fetch existing positions from real brokers
        # to ensure we don't accidentally touch production APIs. We will just pass None.
        existing_position = None

        # Build payload
        signal_input = build_signal_input(
            symbol=symbol,
            asset_class=asset_class,
            account_equity_usd=equity,
            existing_position=existing_position,
            ohlcv_period=config.get("ohlcv_period", "60d"),
            rsi_period=config.get("rsi_period", 14),
            sma_short=config.get("sma_short", 20),
            sma_long=config.get("sma_long", 50),
            max_headlines=config.get("max_headlines", 6),
        )

        # Ask Gemini
        raw_output = generate_signal(
            signal_input=signal_input,
            model="gemini-2.5-flash",
            max_tokens=config.get("max_tokens", 512),
        )

        # Risk Manager validation/override
        # Note: today_realized_loss_pct is 0 for the test as we don't track real PnL
        final_signal = validate(
            raw=raw_output,
            current_price=signal_input.current_price,
            today_realized_loss_pct=0.0,
            circuit_breaker_loss_pct=cb_threshold,
            max_risk_pct=max_risk,
            max_absolute_position_pct=max_abs_pos,
            min_confidence=min_conf,
        )

        # Log it
        bot_logger.log_signal(signal_input, raw_output, final_signal)
        
        # MOCK ExecutionResult (no real execution)
        exec_result = ExecutionResult(
            status="dry_run",
            message="Gemini test dry run - NO EXECUTION",
            realized_pnl_usd=None
        )

        logger.info(
            "[%s] DRY-RUN simulated execution: %s",
            symbol,
            exec_result.message,
        )

    except Exception as exc:
        logger.exception("Failed processing %s: %s", symbol, exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading Bot - Gemini Test Runner")
    parser.add_argument("--symbol", help="Run only for a specific symbol (e.g. BTC-USD)")
    args = parser.parse_args()

    config = load_config()

    # Use a separate database for the Gemini test
    db_path = Path("gemini_test.db")
    bot_logger = BotLogger(db_path=db_path)

    if check_self_termination(bot_logger):
        logger.info("7-day Gemini test period has expired. Self-terminating.")
        sys.exit(0)

    # We use live equity for position sizing parity with production
    try:
        equity = fetch_live_equity(config.get("account_equity_usd", 100.0))
        logger.info("Live equity fetched: $%.2f", equity)
    except Exception as exc:
        logger.error("Failed to fetch live equity: %s. Falling back to config.", exc)
        equity = config.get("account_equity_usd", 100.0)

    symbols_to_run = config.get("symbols", [])
    
    if args.symbol:
        symbols_to_run = [s for s in symbols_to_run if s["symbol"] == args.symbol]
        if not symbols_to_run:
            logger.error("Symbol %s not found in config", args.symbol)
            sys.exit(1)

    for symbol_entry in symbols_to_run:
        symbol = symbol_entry["symbol"]
        asset_class = symbol_entry["asset_class"]
        logger.info("--- Processing %s (%s) ---", symbol, asset_class)
        process_symbol(symbol, asset_class, config, equity, bot_logger)

    logger.info("Cycle complete. Exporting signals to CSV...")
    
    # Export to a separate CSV
    with bot_logger._connect() as conn:
        cursor = conn.execute("SELECT * FROM signals ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        import csv
        with open("gemini_test_signals.csv", "w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.writer(f)
                writer.writerow(rows[0].keys())
                for row in rows:
                    writer.writerow(row)

if __name__ == "__main__":
    main()
