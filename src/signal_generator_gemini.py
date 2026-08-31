"""
signal_generator_gemini.py — Calls the Gemini API with the system prompt defined
in the spec and parses the response into a validated SignalOutput.

The ONLY thing this module does is talk to the model.
All risk enforcement happens in risk_manager.py.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import ValidationError

from .models import SignalInput, SignalOutput
from .signal_generator import SYSTEM_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

# ─── Client factory ───────────────────────────────────────────────────────────

def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file."
        )
    return genai.Client(api_key=api_key)


# ─── Core function ────────────────────────────────────────────────────────────

def generate_signal(
    signal_input: SignalInput,
    model: str = "gemini-2.5-flash",
    max_tokens: int = 512,
    client: Optional[genai.Client] = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> SignalOutput:
    """
    Send signal_input to Gemini and return a validated SignalOutput.

    Raises:
        EnvironmentError: if GEMINI_API_KEY is missing.
        ValueError: if the model returns malformed or unparseable JSON.
        Exception: on API-level failures.
    """
    if client is None:
        client = _get_client()

    user_message = signal_input.model_dump_json(indent=2)

    logger.info("Calling Gemini (%s) for %s …", model, signal_input.symbol)
    logger.debug("Payload:\n%s", user_message)

    try:
        response = client.models.generate_content(
            model=model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            )
        )
    except Exception as e:
        logger.error("Gemini API error: %s", e)
        raise

    if not response.text:
        raise ValueError(f"Model returned empty or null output for {signal_input.symbol}.")
    raw_text = response.text.strip()
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
