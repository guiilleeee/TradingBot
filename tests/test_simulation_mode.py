import pytest
from unittest.mock import patch, MagicMock
from src.main import process_symbol  # type: ignore
from src.logger import BotLogger  # type: ignore
from src.signal_generator import SYSTEM_PROMPT  # type: ignore

@patch("src.main.fetch_existing_position")
@patch("src.main.build_signal_input")
@patch("src.main.generate_signal_gemini")
@patch("src.main.validate")
def test_simulation_vs_live_mode_parameters(
    mock_validate, mock_generate_signal, mock_build_signal_input, mock_fetch_existing_position
):
    """
    Test that live mode ALWAYS uses min_confidence_live=0.55 and the exact original SYSTEM_PROMPT,
    while simulation mode uses min_confidence_simulation and appends the simulation addendum.
    """
    bot_logger = MagicMock(spec=BotLogger)
    bot_logger.get_today_realized_loss_pct.return_value = 0.0

    mock_fetch_existing_position.return_value = None
    mock_build_signal_input.return_value = MagicMock(current_price=100.0)

    # 1. Test is_live = True
    config_live = {
        "live_execution": True,
        "circuit_breaker_loss_pct": 5.0,
        "min_confidence_live": 0.55,
        "min_confidence_simulation": 0.10, # Extraneous value to ensure it's not used
        "signal_provider": "gemini",
    }
    
    process_symbol("BTC-USD", "crypto", config_live, 1000.0, bot_logger, set())

    # Check generate_signal call
    _, kwargs = mock_generate_signal.call_args
    assert kwargs["system_prompt"] == SYSTEM_PROMPT, "Live mode must use exact original SYSTEM_PROMPT."
    
    # Check validate call
    _, kwargs = mock_validate.call_args
    assert kwargs["min_confidence"] == 0.55, "Live mode must use min_confidence_live (0.55)."


    # 2. Test is_live = False
    mock_generate_signal.reset_mock()
    mock_validate.reset_mock()

    config_sim = {
        "live_execution": False,
        "circuit_breaker_loss_pct": 5.0,
        "min_confidence_live": 0.55,
        "min_confidence_simulation": 0.40,
        "signal_provider": "gemini",
    }
    
    process_symbol("BTC-USD", "crypto", config_sim, 1000.0, bot_logger, set())

    # Check generate_signal call
    _, kwargs = mock_generate_signal.call_args
    assert kwargs["system_prompt"] != SYSTEM_PROMPT, "Simulation mode must append addendum to SYSTEM_PROMPT."
    assert "SIMULATION MODE \u2014 ADDITIONAL GUIDANCE" in kwargs["system_prompt"]
    
    # Check validate call
    _, kwargs = mock_validate.call_args
    assert kwargs["min_confidence"] == 0.40, "Simulation mode must use min_confidence_simulation (0.40)."
