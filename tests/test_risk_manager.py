import pytest
from src.models import SignalOutput
from src.risk_manager import validate

@pytest.fixture
def raw_buy() -> SignalOutput:
    return SignalOutput(
        symbol="BTC-USD",
        action="buy",
        confidence=0.8,
        position_size_pct=1.5,
        stop_loss_price=50000.0,
        take_profit_price=70000.0,
        reasoning="Good setup."
    )


def test_no_intervention(raw_buy):
    # current_price = 51000. stop = 50000. dist = 1000/51000 = 0.0196
    # expected_pos = 2.0 / 0.0196 = 102 (capped at 30.0)
    # wait, if I want no intervention except position size adjustment:
    # Actually, the sizing rule ALWAYS interventions to set the size, so it will clamp if we leave max_abs at 30.
    # Let's adjust stop_loss_price so pos_size < 30.
    # dist needed for pos_size < 30: 2 / 30 = 0.066 dist.
    # 55000 * 0.066 = 3666. stop_loss = 51333.
    # Let's just pass a current_price such that pos_size doesn't clamp.
    # dist = 10000/60000 = 0.166. size = 2 / 0.166 = 12.0
    signal = validate(raw_buy, current_price=60000.0, today_realized_loss_pct=2.0)  # +2.0 is a gain
    assert signal.action == "buy"
    assert signal.position_size_pct == 12.0
    assert signal.override_reason is None


def test_circuit_breaker(raw_buy):
    # -6.0% loss exceeds the 5.0% threshold
    signal = validate(raw_buy, current_price=60000.0, today_realized_loss_pct=-6.0, circuit_breaker_loss_pct=5.0)
    assert signal.action == "hold"
    assert signal.position_size_pct == 0.0
    assert signal.stop_loss_price is None
    assert signal.take_profit_price is None
    assert "Circuit breaker" in signal.override_reason
    assert signal.raw_action == "buy"


def test_position_size_cap_tight_stop(raw_buy):
    # current_price = 50500. dist = 500/50500 = 0.0099
    # expected size = 2 / 0.0099 = 202% -> clamped to 30.0
    signal = validate(raw_buy, current_price=50500.0, max_absolute_position_pct=30.0)
    assert signal.action == "buy"
    assert signal.position_size_pct == 30.0  # Clamped
    assert "clamped" in signal.override_reason.lower()


def test_missing_stop_loss(raw_buy):
    raw_buy.stop_loss_price = None
    signal = validate(raw_buy, current_price=60000.0)
    assert signal.action == "hold"
    assert "missing required field(s): stop_loss_price" in signal.override_reason


def test_hold_zeroes_fields():
    raw_hold = SignalOutput(
        symbol="BTC-USD",
        action="hold",
        confidence=0.6,
        position_size_pct=0,
        stop_loss_price=None,
        take_profit_price=None,
        reasoning="Boring market."
    )
    # The Pydantic model validator does the zeroing, but risk_manager enforces it too
    signal = validate(raw_hold, current_price=60000.0)
    assert signal.action == "hold"
    assert signal.position_size_pct == 0.0
    assert signal.override_reason is None

def test_low_confidence_forces_hold(raw_buy):
    raw_buy.confidence = 0.50
    signal = validate(raw_buy, current_price=60000.0, min_confidence=0.55)
    assert signal.action == "hold"
    assert signal.position_size_pct == 0.0
    assert signal.stop_loss_price is None
    assert signal.take_profit_price is None
    assert "Confidence 0.5 below 0.55 threshold." in signal.override_reason

def test_zero_or_negative_position_size_rejected(raw_buy):
    raw_buy.position_size_pct = -1.0
    signal = validate(raw_buy, current_price=60000.0)
    assert signal.action == "hold"
    assert signal.position_size_pct == 0.0
    assert signal.stop_loss_price is None
    assert signal.take_profit_price is None
    assert "Invalid position size: -1.0." in signal.override_reason

def test_too_tight_stop_loss(raw_buy):
    # stop = 50000, current = 50050. dist = 50/50050 = 0.00099 (< 0.003)
    signal = validate(raw_buy, current_price=50050.0)
    assert signal.action == "hold"
    assert signal.position_size_pct == 0.0
    assert "too tight (under 0.3%)" in signal.override_reason
