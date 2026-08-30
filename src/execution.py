"""
execution.py — Executes trades on real brokers (Alpaca, OKX).
"""
import logging
import os
import requests
import ccxt
from typing import Optional, Any, Literal

from .models import TradeSignal, ExecutionResult, ExistingPosition

logger = logging.getLogger(__name__)

ALPACA_BASE_URL = "https://api.alpaca.markets"  # Could also use paper URL here

def fetch_existing_position(symbol: str, asset_class: str, is_live: bool, bot_logger) -> Optional[ExistingPosition]:
    """
    Fetch existing position size and average entry price for a given symbol.
    If is_live is False, read from the simulated portfolio ledger instead.
    """
    if not is_live:
        return bot_logger.get_simulated_position(symbol)

    if asset_class == "equity":
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_API_SECRET")
        if not api_key or not api_secret:
            return None
            
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
            "Content-Type": "application/json"
        }
        try:
            pos_resp = requests.get(f"{ALPACA_BASE_URL}/v2/positions/{symbol}", headers=headers, timeout=10)
            if pos_resp.status_code == 200:
                pos_data = pos_resp.json()
                qty = float(pos_data.get("qty", 0))
                if qty > 0:
                    avg_price = float(pos_data.get("avg_entry_price", 0))
                    if avg_price > 0:
                        return ExistingPosition(qty=qty, avg_entry_price=avg_price)
        except Exception as e:
            logger.warning("Failed to fetch Alpaca positions for %s: %s", symbol, e)
            
    elif asset_class == "crypto":
        api_key = os.getenv("OKX_API_KEY")
        secret_key = os.getenv("OKX_SECRET_KEY")
        passphrase = os.getenv("OKX_PASSPHRASE")
        if not api_key or not secret_key or not passphrase:
            return None
            
        exchange = ccxt.okx({
            "apiKey": api_key,
            "secret": secret_key,
            "password": passphrase,
            "enableRateLimit": True,
        })
        
        okx_symbol = symbol.replace("-USD", "/USDT")
        base_currency = okx_symbol.split('/')[0]
        try:
            balance = exchange.fetch_balance()
            if base_currency in balance:
                raw_qty = balance[base_currency].get('total')
                qty = float(raw_qty) if raw_qty is not None else 0.0
                if qty > 0:
                    avg_price = bot_logger.get_last_buy_price(symbol)
                    if avg_price > 0:
                        return ExistingPosition(qty=qty, avg_entry_price=avg_price)
        except Exception as e:
            logger.warning("Failed to fetch OKX balance for %s: %s", symbol, e)
            
    return None

def execute_alpaca_trade(signal: TradeSignal, current_price: float, live_equity: float, is_live: bool, existing_position: Optional[ExistingPosition]) -> ExecutionResult:
    """Execute US Equities trade on Alpaca."""
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    
    if is_live and (not api_key or not api_secret):
        return ExecutionResult(status="error", message="Missing Alpaca API credentials")

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
        "Content-Type": "application/json"
    }
    
    # 1. Check existing position to prevent duplicate trades or naked sells
    qty_existing = existing_position.qty if existing_position else 0.0
        
    if signal.action == "buy" and qty_existing > 0:
        msg = f"Trade skipped: Existing buy position of {qty_existing} already exists on Alpaca."
        logger.info(msg)
        return ExecutionResult(status="skipped", message=msg)
    elif signal.action == "sell" and qty_existing <= 0:
        msg = f"Trade skipped: Cannot sell {signal.symbol} on Alpaca because no existing long position exists."
        logger.info(msg)
        return ExecutionResult(status="skipped", message=msg)

    # 2. Sizing calculation
    if signal.action == "sell" and existing_position:
        qty = int(existing_position.qty) # Alpaca uses integer shares usually in this bot
    else:
        target_usd = live_equity * (signal.position_size_pct / 100.0)
        qty = int(target_usd / current_price)
    
    if qty <= 0:
        msg = f"Trade skipped: below Alpaca minimum order size (calculated qty {qty} < 1 share)"
        logger.info(msg)
        return ExecutionResult(status="skipped", message=msg)

    # Compute P&L if closing a position
    pnl = None
    if signal.action == "sell" and existing_position:
        pnl = (current_price - existing_position.avg_entry_price) * qty

    if not is_live:
        msg = f"DRY RUN: would have placed {signal.action} {signal.symbol} size={signal.position_size_pct}% (qty={qty}) @ ~{current_price}"
        logger.info(msg)
        return ExecutionResult(status="dry_run", message=msg, realized_pnl_usd=pnl, qty=qty)

    # 3. Build Bracket Order Payload
    payload: dict[str, Any] = {
        "symbol": signal.symbol,
        "qty": str(qty),
        "side": signal.action,
        "type": "market",
        "time_in_force": "day",
    }
    
    # Add bracket legs if both are present and we are opening a position
    if signal.action == "buy" and signal.stop_loss_price and signal.take_profit_price:
        limit_buffer = 0.99
        payload["order_class"] = "bracket"
        payload["take_profit"] = {
            "limit_price": str(round(signal.take_profit_price, 2))
        }
        payload["stop_loss"] = {
            "stop_price": str(round(signal.stop_loss_price, 2)),
            "limit_price": str(round(signal.stop_loss_price * limit_buffer, 2)) # slight limit buffer
        }

    try:
        resp = requests.post(f"{ALPACA_BASE_URL}/v2/orders", headers=headers, json=payload, timeout=10)
        if resp.status_code in (200, 201):
            data = resp.json()
            order_id = data.get("id")
            logger.info("Alpaca order placed: %s (ID: %s)", signal.action, order_id)
            return ExecutionResult(status="success", order_id=order_id, message="Order submitted", realized_pnl_usd=pnl, qty=qty)
        else:
            msg = f"Alpaca API error: {resp.status_code} {resp.text}"
            logger.error(msg)
            return ExecutionResult(status="error", message=msg)
    except Exception as e:
        msg = f"Exception calling Alpaca: {e}"
        logger.error(msg)
        return ExecutionResult(status="error", message=msg)


def execute_okx_trade(signal: TradeSignal, current_price: float, live_equity: float, is_live: bool, existing_position: Optional[ExistingPosition]) -> ExecutionResult:
    """Execute Crypto trade on OKX."""
    api_key = os.getenv("OKX_API_KEY")
    secret_key = os.getenv("OKX_SECRET_KEY")
    passphrase = os.getenv("OKX_PASSPHRASE")

    if is_live and (not api_key or not secret_key or not passphrase):
        return ExecutionResult(status="error", message="Missing OKX API credentials")

    exchange_config: dict[str, Any] = {"enableRateLimit": True}
    if api_key and secret_key and passphrase:
        exchange_config.update({
            "apiKey": api_key,
            "secret": secret_key,
            "password": passphrase,
        })

    exchange = ccxt.okx(exchange_config)  # type: ignore

    # Map symbol from yfinance 'BTC-USD' to CCXT OKX spot 'BTC/USDT'
    okx_symbol = signal.symbol.replace("-USD", "/USDT")

    try:
        exchange.load_markets()
        market = exchange.market(okx_symbol)
    except Exception as e:
        logger.error("Failed to load OKX markets for %s: %s", okx_symbol, e)
        return ExecutionResult(status="error", message="Could not load market data")

    # 1. Check existing position
    base_currency = okx_symbol.split('/')[0]
    qty_existing = existing_position.qty if existing_position else 0.0

    if signal.action == "buy" and qty_existing > 0:
        msg = f"Trade skipped: Existing spot balance of {qty_existing} {base_currency} already exists on OKX."
        logger.info(msg)
        return ExecutionResult(status="skipped", message=msg)
    elif signal.action == "sell" and qty_existing <= 0:
        msg = f"Trade skipped: Cannot sell {okx_symbol} on OKX spot because no existing balance exists."
        logger.info(msg)
        return ExecutionResult(status="skipped", message=msg)

    # 2. Sizing calculation
    if signal.action == "sell" and existing_position:
        qty = existing_position.qty
    else:
        target_usd = live_equity * (signal.position_size_pct / 100.0)
        qty = target_usd / current_price
    
    # Adjust for precision
    amount = exchange.amount_to_precision(okx_symbol, qty)
    amount_f = float(amount) if amount is not None else 0.0

    min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
    if amount_f < (min_amount or 0.0001):
        msg = f"Trade skipped: below OKX minimum order size ({amount_f} < {min_amount})"
        logger.info(msg)
        return ExecutionResult(status="skipped", message=msg)

    # Compute P&L if closing a position
    pnl = None
    if signal.action == "sell" and existing_position:
        pnl = (current_price - existing_position.avg_entry_price) * amount_f

    if not is_live:
        msg = f"DRY RUN: would have placed {signal.action} {okx_symbol} size={signal.position_size_pct}% (qty={amount_f}) @ ~{current_price}"
        logger.info(msg)
        return ExecutionResult(status="dry_run", message=msg, realized_pnl_usd=pnl, qty=amount_f)

    # 3. Execute
    try:
        params = {}
        # OKX supports attachAlgoOrds directly mapped by CCXT using stopLoss/takeProfit dictionary
        if signal.action == "buy":
            if signal.stop_loss_price:
                params['stopLoss'] = {
                    'triggerPrice': exchange.price_to_precision(okx_symbol, signal.stop_loss_price),
                }
            if signal.take_profit_price:
                params['takeProfit'] = {
                    'triggerPrice': exchange.price_to_precision(okx_symbol, signal.take_profit_price),
                }

        side_literal: Literal["buy", "sell"] = "buy" if signal.action == "buy" else "sell"
        order = exchange.create_order(
            symbol=okx_symbol,
            type="market",
            side=side_literal,
            amount=amount_f,
            params=params
        )
        
        logger.info("OKX order placed: %s (ID: %s)", signal.action, order['id'])
        return ExecutionResult(status="success", order_id=order['id'], message="Order submitted", realized_pnl_usd=pnl, qty=amount_f)
        
    except Exception as e:
        msg = f"OKX execution failed: {e}"
        logger.error(msg)
        return ExecutionResult(status="error", message=msg)


def execute_trade(signal: TradeSignal, asset_class: str, current_price: float, live_equity: float, is_live: bool, existing_position: Optional[ExistingPosition]) -> ExecutionResult:
    """Unified routing for execution, wrapped in a blanket try/except."""
    try:
        if asset_class == "crypto":
            return execute_okx_trade(signal, current_price, live_equity, is_live, existing_position)
        elif asset_class == "equity":
            return execute_alpaca_trade(signal, current_price, live_equity, is_live, existing_position)
        else:
            return ExecutionResult(status="error", message=f"Unknown asset class: {asset_class}")
    except Exception as e:
        logger.exception("Catastrophic error in execute_trade for %s: %s", signal.symbol, e)
        return ExecutionResult(status="error", message="Unhandled exception in execution layer")
