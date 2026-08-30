"""
Pydantic v2 data models for TradingBot.

These are the single source of truth for the shape of data flowing
between data_fetcher → signal_generator → risk_manager → logger.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Execution ───────────────────────────────────────────────────────────────

class ExecutionResult(BaseModel):
    status: str  # "success", "skipped", "error", "dry_run"
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    message: Optional[str] = None
    realized_pnl_usd: Optional[float] = None
    qty: Optional[float] = None


# ─── Input to Claude ─────────────────────────────────────────────────────────

class TechnicalIndicators(BaseModel):
    rsi_14: float = Field(..., ge=0, le=100)
    sma_20: float = Field(..., gt=0)
    sma_50: float = Field(..., gt=0)
    price_change_24h_pct: float
    volume_change_24h_pct: float


class ExistingPosition(BaseModel):
    qty: float
    avg_entry_price: float = Field(..., gt=0)


class SignalInput(BaseModel):
    symbol: str
    asset_class: str = Field(..., pattern="^(equity|crypto)$")
    current_price: float = Field(..., gt=0)
    account_equity_usd: float = Field(..., gt=0)
    existing_position: Optional[ExistingPosition] = None
    technical_indicators: TechnicalIndicators
    recent_headlines: list[str] = Field(default_factory=list)


# ─── Raw output from Claude (before risk-manager post-processing) ─────────────

class SignalOutput(BaseModel):
    symbol: str
    action: str = Field(..., pattern="^(buy|sell|hold)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    position_size_pct: float = Field(..., ge=0.0, le=100.0)  # model may return bad value; risk_manager clamps
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    reasoning: str

    @field_validator("action")
    @classmethod
    def action_lowercase(cls, v: str) -> str:
        return v.lower().strip()

    @model_validator(mode="after")
    def hold_nulls_prices(self) -> "SignalOutput":
        """If hold, both prices must be null."""
        if self.action == "hold":
            self.stop_loss_price = None
            self.take_profit_price = None
            self.position_size_pct = 0.0
        return self


# ─── Final signal after risk-manager validation ───────────────────────────────

class TradeSignal(BaseModel):
    """
    The signal that should actually be acted on (or logged).
    override_reason is non-None whenever risk_manager changed the raw output.
    """
    symbol: str
    action: str
    confidence: float
    position_size_pct: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    reasoning: str
    override_reason: Optional[str] = None   # set by risk_manager when it intervenes
    raw_action: str                          # model's original action before any override
