"""
risk_manager.py — Enforces the four non-negotiable safety rules in Python.

The model is a black box; we never trust it blindly.
Every rule that fires sets override_reason on the returned TradeSignal.
"""
from __future__ import annotations

import logging
from typing import Optional

from .models import SignalOutput, TradeSignal

logger = logging.getLogger(__name__)
def validate(
    raw: SignalOutput,
    current_price: float,
    today_realized_loss_pct: float = 0.0,
    circuit_breaker_loss_pct: float = 5.0,
    max_risk_pct: float = 2.0,
    max_absolute_position_pct: float = 30.0,
    min_confidence: float = 0.55,
) -> TradeSignal:
    """
    Apply all risk rules to a raw SignalOutput and return a safe TradeSignal.

    Parameters
    ----------
    raw : SignalOutput
        The unmodified output from signal_generator.generate_signal().
    today_realized_loss_pct : float
        Realized P&L for today as a percentage of account equity (negative = loss).
        Provided by logger.get_today_realized_loss_pct().
    circuit_breaker_loss_pct : float
        If today's loss exceeds this threshold (positive number), halt trading.
    max_risk_pct : float
        Maximum percentage of equity to risk on a single trade.
    max_absolute_position_pct : float
        Hard cap on total position size as a percentage of equity.
    min_confidence : float
        Minimum confidence to take a trade. Below this, action is overridden to hold.

    Returns
    -------
    TradeSignal
        Final signal ready to be acted on. override_reason is set when any
        rule was triggered.
    """
    action = raw.action
    position_size_pct = raw.position_size_pct
    stop_loss_price = raw.stop_loss_price
    take_profit_price = raw.take_profit_price
    override_reasons: list[str] = []

    # ── Rule 1: Daily loss circuit breaker ───────────────────────────────────
    # today_realized_loss_pct is negative when there are losses (e.g. -6.0 means -6%)
    if today_realized_loss_pct <= -circuit_breaker_loss_pct:
        if action in ("buy", "sell"):
            logger.warning(
                "CIRCUIT BREAKER: today's realized loss (%.2f%%) exceeds threshold (%.2f%%). "
                "Overriding %s → hold.",
                abs(today_realized_loss_pct),
                circuit_breaker_loss_pct,
                action,
            )
            override_reasons.append(
                f"Circuit breaker: daily realized loss ({abs(today_realized_loss_pct):.2f}%) "
                f"exceeds {circuit_breaker_loss_pct:.2f}% threshold."
            )
            action = "hold"
            position_size_pct = 0.0
            stop_loss_price = None
            take_profit_price = None

    # ── Rule 2: Confidence threshold ─────────────────────────────────────────
    if action != "hold" and raw.confidence < min_confidence:
        logger.warning(
            "LOW CONFIDENCE: model confidence %.2f is below %.2f threshold for %s. "
            "Overriding %s → hold.",
            raw.confidence,
            min_confidence,
            raw.symbol,
            action,
        )
        override_reasons.append(
            f"Confidence {raw.confidence} below {min_confidence} threshold."
        )
        action = "hold"
        position_size_pct = 0.0
        stop_loss_price = None
        take_profit_price = None

    # ── Rule 3: Invalid position size ────────────────────────────────────────
    if action in ("buy", "sell") and position_size_pct <= 0:
        logger.warning(
            "INVALID POSITION SIZE: model returned %.2f%% for %s. Overriding %s → hold.",
            position_size_pct,
            raw.symbol,
            action,
        )
        override_reasons.append(
            f"Invalid position size: {position_size_pct}."
        )
        action = "hold"
        position_size_pct = 0.0
        stop_loss_price = None
        take_profit_price = None

    # ── Rule 4: Risk-based position sizing ───────────────────────────────────
    if action in ("buy", "sell") and stop_loss_price is not None:
        stop_distance_pct = abs(current_price - stop_loss_price) / current_price
        
        if stop_distance_pct < 0.003:
            logger.warning(
                "TIGHT STOP: %s stop-loss distance %.4f%% is too tight (under 0.3%%). "
                "Overriding %s → hold.",
                raw.symbol,
                stop_distance_pct * 100,
                action,
            )
            override_reasons.append(
                f"Stop-loss too tight (under 0.3%). Overriding to hold."
            )
            action = "hold"
            position_size_pct = 0.0
            stop_loss_price = None
            take_profit_price = None
        else:
            # position_size_pct_of_equity = max_risk_pct / stop_distance_pct
            computed_position_size = max_risk_pct / stop_distance_pct
            
            if computed_position_size > max_absolute_position_pct:
                logger.warning(
                    "POSITION SIZE CAP: risk-based size %.2f%% for %s clamped to absolute limit %.2f%%.",
                    computed_position_size,
                    raw.symbol,
                    max_absolute_position_pct,
                )
                override_reasons.append(
                    f"Position size clamped from {computed_position_size:.2f}% to {max_absolute_position_pct:.2f}%."
                )
                position_size_pct = max_absolute_position_pct
            else:
                position_size_pct = computed_position_size

    # ── Rule 5: Reject buy/sell without stop-loss and take-profit ────────────
    if action in ("buy", "sell"):
        missing: list[str] = []
        if stop_loss_price is None:
            missing.append("stop_loss_price")
        if take_profit_price is None:
            missing.append("take_profit_price")

        if missing:
            logger.warning(
                "MISSING PRICES: %s signal for %s is missing %s. Overriding → hold.",
                action,
                raw.symbol,
                ", ".join(missing),
            )
            override_reasons.append(
                f"Rejected {action}: missing required field(s): {', '.join(missing)}. "
                "Defaulting to hold."
            )
            action = "hold"
            position_size_pct = 0.0
            stop_loss_price = None
            take_profit_price = None

    # ── Rule 6: Ensure hold zeroes everything out ────────────────────────────
    if action == "hold":
        position_size_pct = 0.0
        stop_loss_price = None
        take_profit_price = None

    combined_reason: Optional[str] = "; ".join(override_reasons) if override_reasons else None

    return TradeSignal(
        symbol=raw.symbol,
        action=action,
        confidence=raw.confidence,
        position_size_pct=position_size_pct,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        reasoning=raw.reasoning,
        override_reason=combined_reason,
        raw_action=raw.action,
    )
