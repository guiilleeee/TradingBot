import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.logger import BotLogger
from src.main import process_symbol

def test_circuit_breaker_logic(tmp_path: Path):
    """
    Test that the circuit breaker correctly triggers and skips API calls
    when today's realized loss exceeds the threshold.
    """
    # 1. Setup in-memory BotLogger (using tmp_path for the DB file)
    db_path = tmp_path / "test_trading_bot.db"
    bot_logger = BotLogger(db_path=db_path)
    
    symbol = "TEST-USD"
    asset_class = "crypto"
    equity = 1000.0
    
    # Configure circuit breaker threshold to 5%
    config = {
        "circuit_breaker_loss_pct": 5.0,
        "max_position_size_pct": 10.0,
        "min_confidence": 0.55
    }
    
    # 2. Insert a realized loss of $60 (which is a 6% loss on $1000 equity)
    bot_logger.record_pnl(symbol, -60.0)
    
    # 3. Confirm get_today_realized_loss_pct reflects the loss
    loss_pct = bot_logger.get_today_realized_loss_pct(equity)
    assert loss_pct == -6.0, f"Expected loss to be -6.0%, got {loss_pct}%"
    
    # 4. Confirm circuit breaker fires on the next cycle
    # We patch fetch_existing_position and generate_signal to ensure they are NOT called
    with patch("src.main.fetch_existing_position") as mock_fetch, \
         patch("src.main.build_signal_input") as mock_build, \
         patch("src.main.generate_signal") as mock_generate:
        
        # Run process_symbol
        process_symbol(symbol, asset_class, config, equity, bot_logger)
        
        # Assert that execution was skipped (no API calls made)
        mock_fetch.assert_not_called()
        mock_build.assert_not_called()
        mock_generate.assert_not_called()
