"""
logger.py — SQLite-backed logging for every signal and realized P&L entry.

Schema
------
signals : one row per signal cycle
    id, timestamp, symbol, signal_input_json, raw_output_json,
    final_signal_json, override_reason

pnl : filled manually or by execution layer
    id, timestamp, symbol, realized_pnl_usd

Usage
-----
    from src.logger import BotLogger
    bot_logger = BotLogger()
    bot_logger.log_signal(signal_input, raw_output, final_signal)
    loss_pct = bot_logger.get_today_realized_loss_pct(account_equity_usd=500)
"""
from __future__ import annotations

import csv
import json
import logging
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from .models import SignalInput, SignalOutput, TradeSignal, ExecutionResult

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("trading_bot.db")


class BotLogger:
    """Thread-safe-ish SQLite logger (single-writer assumed)."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS signals (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT NOT NULL,
                    symbol          TEXT NOT NULL,
                    signal_input    TEXT NOT NULL,   -- JSON
                    raw_output      TEXT NOT NULL,   -- JSON
                    final_signal    TEXT NOT NULL,   -- JSON
                    override_reason TEXT,            -- NULL if no override
                    execution_result TEXT            -- JSON, NULL if hold
                );

                CREATE TABLE IF NOT EXISTS pnl (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp           TEXT NOT NULL,
                    symbol              TEXT NOT NULL,
                    realized_pnl_usd    REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS simulated_positions (
                    symbol            TEXT PRIMARY KEY,
                    qty               REAL NOT NULL,
                    avg_entry_price   REAL NOT NULL,
                    stop_loss_price   REAL,
                    take_profit_price REAL,
                    opened_at         TEXT NOT NULL
                );
            """)
            
            # Simple schema evolution: add new columns if they don't exist
            try:
                conn.execute("ALTER TABLE signals ADD COLUMN execution_result TEXT")
            except sqlite3.OperationalError:
                pass

        logger.debug("Database initialised at %s", self.db_path.resolve())

    # ── Signal logging ────────────────────────────────────────────────────────

    def log_signal(
        self,
        signal_input: SignalInput,
        raw_output: SignalOutput,
        final_signal: TradeSignal,
        execution_result: Optional[ExecutionResult] = None,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO signals
                    (timestamp, symbol, signal_input, raw_output, final_signal, override_reason, execution_result)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    final_signal.symbol,
                    signal_input.model_dump_json(),
                    raw_output.model_dump_json(),
                    final_signal.model_dump_json(),
                    final_signal.override_reason,
                    execution_result.model_dump_json() if execution_result else None,
                ),
            )
        logger.info(
            "Logged signal for %s: action=%s override=%s",
            final_signal.symbol,
            final_signal.action,
            final_signal.override_reason or "none",
        )

    def log_auto_close_signal(self, symbol: str, reason: str, price: float, qty: float, pnl: float) -> None:
        """Log a synthetic 'sell' signal for auto-closed positions so the dashboard sees it."""
        ts = datetime.now(timezone.utc).isoformat()
        
        # Build minimal mock JSONs for the dashboard
        signal_input_json = json.dumps({"symbol": symbol, "current_price": price})
        raw_output_json = json.dumps({"action": "sell", "reasoning": reason})
        final_signal_json = json.dumps({"action": "sell", "symbol": symbol, "override_reason": reason})
        exec_result_json = json.dumps({"status": "success", "realized_pnl_usd": pnl, "qty": qty, "message": reason})
        
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO signals
                    (timestamp, symbol, signal_input, raw_output, final_signal, override_reason, execution_result)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, symbol, signal_input_json, raw_output_json, final_signal_json, reason, exec_result_json),
            )
        logger.info("Logged synthetic auto-close signal for %s: %s", symbol, reason)

    # ── P&L helpers ──────────────────────────────────────────────────────────

    def record_pnl(self, symbol: str, realized_pnl_usd: float) -> None:
        """Call this from your execution layer when a trade closes."""
        ts = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO pnl (timestamp, symbol, realized_pnl_usd) VALUES (?, ?, ?)",
                (ts, symbol, realized_pnl_usd),
            )
        logger.info("Recorded P&L for %s: %.2f USD", symbol, realized_pnl_usd)

    def get_today_realized_loss_pct(self, account_equity_usd: float) -> float:
        """
        Return today's total realized P&L as a percentage of account equity.
        Negative means a net loss. Zero if no trades recorded today.
        """
        today_str = datetime.now(timezone.utc).date().isoformat()  # e.g. "2025-08-28"
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl_usd), 0) as total "
                "FROM pnl WHERE timestamp LIKE ?",
                (f"{today_str}%",),
            ).fetchone()
        total_usd: float = row["total"] if row else 0.0
        pct = (total_usd / account_equity_usd) * 100.0
        logger.debug(
            "Today's realized P&L: %.2f USD (%.2f%% of %.2f equity)",
            total_usd, pct, account_equity_usd,
        )
        return round(pct, 4)

    def get_last_buy_price(self, symbol: str) -> float:
        """Find the execution price of the most recent 'buy' signal for this symbol."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT execution_result, signal_input
                FROM signals
                WHERE symbol = ? AND json_extract(final_signal, '$.action') = 'buy'
                ORDER BY id DESC LIMIT 1
                """,
                (symbol,)
            ).fetchone()
        if not row:
            return 0.0
            
        try:
            if row["execution_result"]:
                er = json.loads(row["execution_result"])
                if er.get("fill_price"):
                    return float(er["fill_price"])
            
            # Fallback to the current_price at the time of the signal
            si = json.loads(row["signal_input"])
            return float(si.get("current_price", 0.0))
        except Exception as e:
            logger.warning("Error parsing last buy price for %s: %s", symbol, e)
            return 0.0

    # ── Simulated ledger (dry run) ────────────────────────────────────────────

    def get_simulated_position(self, symbol: str) -> Optional[ExistingPosition]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT qty, avg_entry_price FROM simulated_positions WHERE symbol = ?",
                (symbol,)
            ).fetchone()
        if row:
            return ExistingPosition(qty=row["qty"], avg_entry_price=row["avg_entry_price"])
        return None

    def get_all_simulated_positions(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM simulated_positions").fetchall()
        return [dict(r) for r in rows]

    def open_simulated_position(
        self, symbol: str, qty: float, price: float, stop_loss_price: Optional[float], take_profit_price: Optional[float]
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO simulated_positions
                    (symbol, qty, avg_entry_price, stop_loss_price, take_profit_price, opened_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    qty = excluded.qty,
                    avg_entry_price = excluded.avg_entry_price,
                    stop_loss_price = excluded.stop_loss_price,
                    take_profit_price = excluded.take_profit_price,
                    opened_at = excluded.opened_at
                """,
                (symbol, qty, price, stop_loss_price, take_profit_price, ts),
            )
        logger.info("Simulated position opened for %s", symbol)

    def close_simulated_position(self, symbol: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM simulated_positions WHERE symbol = ?", (symbol,))
        logger.info("Simulated position closed for %s", symbol)

    # ── CSV export ────────────────────────────────────────────────────────────

    def export_signals_csv(self, output_path: Path) -> int:
        """
        Export the signals table to a flat CSV for offline analysis.
        Returns the number of rows exported.
        """
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM signals ORDER BY id").fetchall()

        if not rows:
            logger.info("No signals to export.")
            return 0

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows([dict(r) for r in rows])

        logger.info("Exported %d signal rows to %s", len(rows), output_path)
        return len(rows)

    def get_signal_stats(self) -> dict:
        """Quick summary stats for the logged signals (useful for edge review)."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            breakdown = conn.execute(
                "SELECT json_extract(final_signal, '$.action') as action, COUNT(*) as cnt "
                "FROM signals GROUP BY action"
            ).fetchall()
            overridden = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE override_reason IS NOT NULL"
            ).fetchone()[0]

        action_counts = {row["action"]: row["cnt"] for row in breakdown}
        return {
            "total_signals": total,
            "action_breakdown": action_counts,
            "overridden_by_risk_manager": overridden,
        }
