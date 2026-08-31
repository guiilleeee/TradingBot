"""
main.py — The entry point and execution loop.

Usage:
  python -m src.main                    # Run one cycle for all config symbols
  python -m src.main --symbol BTC-USD   # Run one cycle for one symbol
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .account_manager import fetch_live_equity
from .data_fetcher import build_signal_input, fetch_ohlcv, get_current_price
from .logger import BotLogger
from .risk_manager import validate
from .signal_generator import generate_signal
from .signal_generator_gemini import generate_signal as generate_signal_gemini
from .execution import execute_trade, fetch_existing_position

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("src.main")


def load_config(path: Path = Path("config.yaml")) -> dict[str, Any]:
    if not path.exists():
        logger.error("Config file not found at %s", path)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def process_symbol(symbol: str, asset_class: str, config: dict[str, Any], equity: float, bot_logger: BotLogger, auto_closed_symbols: set[str]) -> None:
    """End-to-end flow for a single symbol."""
    cb_threshold = config["circuit_breaker_loss_pct"]
    max_risk = config.get("max_risk_pct", 2.0)
    max_abs_pos = config.get("max_absolute_position_pct", 30.0)

    if symbol in auto_closed_symbols:
        logger.info("Skipping model call for %s: auto-closed this cycle.", symbol)
        return

    try:
        # 1. Check circuit breaker status first (to save API calls if already busted)
        today_loss_pct = bot_logger.get_today_realized_loss_pct(equity)
        if today_loss_pct <= -cb_threshold:
            logger.warning(
                "Skipping API call for %s: Circuit breaker tripped (%.2f%% loss >= %.2f%%).",
                symbol, abs(today_loss_pct), cb_threshold
            )
            return

        is_live = config.get("live_execution", False)

        min_conf = config.get("min_confidence_live", 0.55) if is_live else config.get("min_confidence_simulation", 0.40)

        # 1.5 Fetch real position from broker
        existing_position = fetch_existing_position(symbol, asset_class, is_live, bot_logger)

        # 2. Build payload
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

        # 3. Ask Model
        from .signal_generator import SYSTEM_PROMPT
        effective_system_prompt = SYSTEM_PROMPT
        if not is_live:
            effective_system_prompt += (
                "\n\nSIMULATION MODE — ADDITIONAL GUIDANCE (this block only applies when no real capital is\n"
                "at risk):\n"
                "You may weigh moderate-strength setups more actively than you would in live trading —\n"
                "act on directionally reasonable technical alignment even without the clearest possible\n"
                "confirmation, as long as you still provide a genuine, well-placed stop-loss and\n"
                "take-profit and state your true confidence honestly.\n"
                "Do not fabricate confidence or invent a rationale that isn't supported by the data — if\n"
                "a setup genuinely looks bad or contradictory, the correct call is still hold. This\n"
                "guidance widens what counts as \"good enough to act on,\" it does not change what \"good\"\n"
                "means."
            )
            
        provider = config.get("signal_provider", "gemini")
        if provider == "claude":
            raw_output = generate_signal(
                signal_input=signal_input,
                model=config.get("model", "claude-haiku-4-5-20251001"),
                max_tokens=config.get("max_tokens", 512),
                system_prompt=effective_system_prompt,
            )
        elif provider == "gemini":
            raw_output = generate_signal_gemini(
                signal_input=signal_input,
                model=config.get("gemini_model", "gemini-3.7-flash"),
                max_tokens=config.get("max_tokens", 512),
                system_prompt=effective_system_prompt,
            )
        else:
            logger.error("Unknown signal provider: %s", provider)
            return

        # 4. Risk Manager validation/override
        final_signal = validate(
            raw=raw_output,
            current_price=signal_input.current_price,
            today_realized_loss_pct=today_loss_pct,
            circuit_breaker_loss_pct=cb_threshold,
            max_risk_pct=max_risk,
            max_absolute_position_pct=max_abs_pos,
            min_confidence=min_conf,
        )

        # 4.5 Execution
        exec_result = None
        if final_signal.action != "hold":
            exec_result = execute_trade(
                signal=final_signal,
                asset_class=asset_class,
                current_price=signal_input.current_price,
                live_equity=equity,
                is_live=is_live,
                existing_position=existing_position
            )

        # 5. Log it and Record PnL
        bot_logger.log_signal(signal_input, raw_output, final_signal, exec_result)
        
        if exec_result and exec_result.status in ("success", "dry_run"):
            if exec_result.realized_pnl_usd is not None:
                bot_logger.record_pnl(symbol, exec_result.realized_pnl_usd)
            
            # Post-trade Simulated Ledger Maintenance
            if not is_live and exec_result.status == "dry_run":
                if final_signal.action == "buy" and exec_result.qty:
                    bot_logger.open_simulated_position(
                        symbol=symbol,
                        qty=exec_result.qty,
                        price=signal_input.current_price,
                        stop_loss_price=final_signal.stop_loss_price,
                        take_profit_price=final_signal.take_profit_price
                    )
                elif final_signal.action == "sell":
                    bot_logger.close_simulated_position(symbol)

    except Exception as exc:
        logger.exception("Error processing symbol %s: %s", symbol, exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading Signal Bot")
    parser.add_argument("--symbol", type=str, help="Run only for this symbol")
    args = parser.parse_args()

    config = load_config()
    bot_logger = BotLogger()

    # Filter symbols if a specific one was requested
    symbols_to_run = config["symbols"]
    if args.symbol:
        args.symbol = args.symbol.upper()
        symbols_to_run = [s for s in symbols_to_run if s["symbol"].upper() == args.symbol]
        if not symbols_to_run:
            logger.error("Symbol '%s' not found in config.yaml", args.symbol)
            sys.exit(1)

    is_live = config.get("live_execution", False)
    fallback_equity = config.get("account_equity_usd", 100.0)

    # Fetch real live equity or compute simulated equity
    if is_live:
        equity = fetch_live_equity(fallback_equity)
    else:
        equity = fallback_equity + bot_logger.get_all_time_realized_pnl()

    auto_closed_symbols = set()

    # Pre-cycle sweep for simulated stop-loss/take-profit
    if not is_live:
        unrealized_pnl = 0.0
        for pos in bot_logger.get_all_simulated_positions():
            sym = pos["symbol"]
            try:
                # Fetch recent price
                df = fetch_ohlcv(sym, period="5d")
                curr_price = get_current_price(df)
            except Exception as e:
                logger.error("Failed to fetch price for sweep on %s: %s", sym, e)
                continue
            
            sl = pos["stop_loss_price"]
            tp = pos["take_profit_price"]
            
            if (sl and curr_price <= sl) or (tp and curr_price >= tp):
                reason = "Auto-closed: stop-loss hit" if (sl and curr_price <= sl) else "Auto-closed: take-profit hit"
                pnl = (curr_price - pos["avg_entry_price"]) * pos["qty"]
                
                bot_logger.record_pnl(sym, pnl)
                bot_logger.close_simulated_position(sym)
                
                # Update local equity so this closure reflects in the signal log
                equity += pnl
                bot_logger.log_auto_close_signal(sym, reason, curr_price, pos["qty"], pnl, equity)
                
                auto_closed_symbols.add(sym)
            else:
                # Position stays open, accumulate unrealized P&L
                unrealized_pnl += (curr_price - pos["avg_entry_price"]) * pos["qty"]
        
        equity += unrealized_pnl

    logger.info("Starting signal cycle for %d symbols...", len(symbols_to_run))
    for sym_data in symbols_to_run:
        process_symbol(sym_data["symbol"], sym_data["asset_class"], config, equity, bot_logger, auto_closed_symbols)
    
    logger.info("Cycle complete. Exporting CSV logs for dashboard...")
    bot_logger.export_signals_csv(Path("signals.csv"))
    
    # Also dump basic stats to JSON so the dashboard can render quick numbers if needed
    stats = bot_logger.get_signal_stats()
    logger.info("Session Stats: %s", stats)


if __name__ == "__main__":
    main()
