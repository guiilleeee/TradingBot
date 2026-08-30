"""
signal_generator.py — Calls the Claude API with the system prompt defined
in the spec and parses the response into a validated SignalOutput.

The ONLY thing this module does is talk to the model.
All risk enforcement happens in risk_manager.py.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import anthropic
from dotenv import load_dotenv
from pydantic import ValidationError

from .models import SignalInput, SignalOutput

load_dotenv()
logger = logging.getLogger(__name__)

# ─── System prompt (verbatim from the spec) ───────────────────────────────────

SYSTEM_PROMPT = """You are a disciplined quantitative trading signal analyst. You analyze market data
and produce a single trading recommendation in strict JSON format. You are one
component in a larger system — a separate risk-management layer will validate and
can override your output, so your job is to be accurate and honest, not persuasive.

HARD RULES (never violate these):
1. Position sizing is handled downstream based on your stop-loss distance. Do not worry about sizing, focus on setting a sensible, well-reasoned stop-loss.
2. If action is "buy" or "sell", you MUST include a stop_loss_price and a
   take_profit_price. If action is "hold", both must be null.
3. Default to "hold" whenever signals conflict, data is incomplete, or your
   confidence would be below 0.55. Do not force a trade to seem useful.
4. Do not let recent price pumps alone drive a "buy" — momentum without
   confirming volume or news context is a weak signal, say so explicitly.
5. Treat news/headlines as directional bias only, never as certainty.
6. State your confidence honestly. Most days the right answer is "hold".
7. Never assume leverage is available. All position sizing is spot, unencumbered.

INPUT FORMAT (JSON):
{
  "symbol": string,
  "asset_class": "equity" | "crypto",
  "current_price": number,
  "account_equity_usd": number,
  "existing_position": { "qty": number, "avg_entry_price": number } | null,
  "technical_indicators": {
    "rsi_14": number,
    "sma_20": number,
    "sma_50": number,
    "price_change_24h_pct": number,  // Note: change since the last daily bar (may be ~3 days over a weekend for equities)
    "volume_change_24h_pct": number
  },
  "recent_headlines": [string, ...]
}

OUTPUT FORMAT (JSON only, no other text):
{
  "symbol": string,
  "action": "buy" | "sell" | "hold",
  "confidence": number,
  "position_size_pct": number,
  "stop_loss_price": number | null,
  "take_profit_price": number | null,
  "reasoning": string
}

Respond with ONLY the JSON object. No markdown, no commentary outside the JSON."""


# ─── Client factory ───────────────────────────────────────────────────────────

def _get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=api_key)


# ─── Core function ────────────────────────────────────────────────────────────

def generate_signal(
    signal_input: SignalInput,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 512,
    client: Optional[anthropic.Anthropic] = None,
) -> SignalOutput:
    """
    Send signal_input to Claude and return a validated SignalOutput.

    Raises:
        EnvironmentError: if ANTHROPIC_API_KEY is missing.
        ValueError: if the model returns malformed or unparseable JSON.
        anthropic.APIError: on API-level failures (rate limits, server errors).
    """
    if client is None:
        client = _get_client()

    user_message = signal_input.model_dump_json(indent=2)

    logger.info("Calling Claude (%s) for %s …", model, signal_input.symbol)
    logger.debug("Payload:\n%s", user_message)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = response.content[0].text.strip()
    logger.debug("Raw model response:\n%s", raw_text)

    # Strip accidental markdown fences the model might add despite instructions
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        raw_text = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    try:
        raw_dict = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model returned non-JSON output for {signal_input.symbol}.\n"
            f"Raw text: {raw_text!r}\nOriginal error: {exc}"
        ) from exc

    try:
        output = SignalOutput.model_validate(raw_dict)
    except ValidationError as exc:
        raise ValueError(
            f"Model output failed schema validation for {signal_input.symbol}.\n"
            f"Errors: {exc}"
        ) from exc

    # Sanity-check: symbol should match (model might hallucinate a different one)
    if output.symbol.upper() != signal_input.symbol.upper():
        logger.warning(
            "Symbol mismatch: expected %s, model returned %s. Correcting.",
            signal_input.symbol,
            output.symbol,
        )
        output.symbol = signal_input.symbol

    logger.info(
        "Signal for %s: action=%s confidence=%.2f position_size_pct=%.2f",
        output.symbol, output.action, output.confidence, output.position_size_pct,
    )
    return output
