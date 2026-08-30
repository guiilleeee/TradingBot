import json
from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError

from src.models import SignalInput, TechnicalIndicators
from src.signal_generator import generate_signal


@pytest.fixture
def dummy_input() -> SignalInput:
    return SignalInput(
        symbol="BTC-USD",
        asset_class="crypto",
        current_price=60000.0,
        account_equity_usd=1000.0,
        technical_indicators=TechnicalIndicators(
            rsi_14=45.0,
            sma_20=59000.0,
            sma_50=55000.0,
            price_change_24h_pct=2.5,
            volume_change_24h_pct=-10.0
        )
    )

def test_generate_signal_success(dummy_input):
    mock_client = MagicMock()
    # Mock the anthropic response structure
    mock_message = MagicMock()
    mock_message.content = [
        MagicMock(text=json.dumps({
            "symbol": "BTC-USD",
            "action": "buy",
            "confidence": 0.8,
            "position_size_pct": 1.5,
            "stop_loss_price": 55000.0,
            "take_profit_price": 65000.0,
            "reasoning": "Looks good."
        }))
    ]
    mock_client.messages.create.return_value = mock_message

    output = generate_signal(dummy_input, client=mock_client)
    
    assert output.symbol == "BTC-USD"
    assert output.action == "buy"
    assert output.confidence == 0.8
    assert output.position_size_pct == 1.5

def test_generate_signal_malformed_json(dummy_input):
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="This is not JSON")]
    mock_client.messages.create.return_value = mock_message

    with pytest.raises(ValueError, match="Model returned non-JSON output"):
        generate_signal(dummy_input, client=mock_client)

def test_generate_signal_schema_validation_failure(dummy_input):
    mock_client = MagicMock()
    mock_message = MagicMock()
    # Missing required 'reasoning' field
    mock_message.content = [
        MagicMock(text=json.dumps({
            "symbol": "BTC-USD",
            "action": "buy",
            "confidence": 0.8,
            "position_size_pct": 1.5,
            "stop_loss_price": 55000.0,
            "take_profit_price": 65000.0
        }))
    ]
    mock_client.messages.create.return_value = mock_message

    with pytest.raises(ValueError, match="failed schema validation"):
        generate_signal(dummy_input, client=mock_client)
