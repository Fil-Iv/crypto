# trade_manager.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from .logger import log
from .notifier import send_telegram
from .trade_logger import log_trade

# Optional storage hooks
try:
    from .utils.storage import insert_order
except Exception:  # pragma: no cover
    def insert_order(*args, **kwargs):
        return None

# Access DRY_RUN from exchange factory if present
try:
    from .utils.exchange_factory import DRY_RUN  # type: ignore
except Exception:
    DRY_RUN = False

# ---------------------------- Config -----------------------------------------

def _get_env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default

TRAIL_PCT_DEFAULT = _get_env_float("TRAIL_PCT", 0.01)   # 1% trailing stop
PARTIAL_PCT_DEFAULT = _get_env_float("PARTIAL_PCT", 0.5)  # 50% size at first TP
TP1_FRACTION_DEFAULT = _get_env_float("TP1_FRACTION", 0.5) # first TP at 50% of TP

# ---------------------------- Data -------------------------------------------

@dataclass
class Bracket:
    entry: float
    tp: float          # e.g. 0.02 = +2%
    sl: float          # e.g. 0.01 = -1%
    trail_pct: float   # trailing stop percent (e.g. 0.01 = 1%)
    partial_pct: float # fraction to close on first TP stage
    tp1_fraction: float # TP1 at tp1_fraction * tp
    amount_total: float
    amount_open: float
    highest: float     # for trailing stop
    lowest: float      # reserved for short side (not used now)
    tp1_done: bool = False

# ---------------------------- TradeManager -----------------------------------

class TradeManager:
    """Execution layer: market orders + bracket management (TP/SL/trailing/partial)."""
    def __init__(self):
        self.brackets: Dict[str, Bracket] = {}

    # -------------------- helpers --------------------
    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _guard_amount(self, exchange, symbol: str, amount: float) -> float:
        """Round amount to exchange precision and ensure min notional."""
        try:
            price = float(exchange.fetch_ticker(symbol).get("last", 0.0))
        except Exception:
            price = 0.0
        prec = 6
        min_cost = 5.0
        try:
            prec = int(exchange.amount_precision(symbol))
            min_cost = float(exchange.min_notional(symbol))
        except Exception:
            pass
        amt = round(max(0.0, amount), prec)
        if price > 0 and amt * price < min_cost:
            amt = round((min_cost / price) * 1.01, prec)
        return amt

    # -------------------- position bookkeeping --------
    def open_position(self, symbol: str, entry_price: float, amount: float,
                      tp: float, sl: float,
                      trail_pct: Optional[float] = None,
                      partial_pct: Optional[float] = None,
                      tp1_fraction: Optional[float] = None) -> None:
        br = Bracket(
            entry=float(entry_price),
            tp=float(tp),
            sl=float(sl),
            trail_pct=float(trail_pct if trail_pct is not None else TRAIL_PCT_DEFAULT),
            partial_pct=float(partial_pct if partial_pct is not None else PARTIAL_PCT_DEFAULT),
            tp1_fraction=float(tp1_fraction if tp1_fraction is not None else TP1_FRACTION_DEFAULT),
            amount_total=float(amount),
            amount_open=float(amount),
            highest=float(entry_price),
            lowest=float(entry_price),
            tp1_done=False,
        )
        self.brackets[symbol] = br
        try:
            log(f"[brackets/open] {symbol} entry={entry_price} tp={tp} sl={sl} trail={br.trail_pct} amt={amount}")
        except Exception:
            pass

    def close_position(self, exchange, symbol: str, reason: str) -> None:
        br = self.brackets.get(symbol)
        if not br or br.amount_open <= 0:
            return
        amt = br.amount_open
        try:
            amt = self._guard_amount(exchange, symbol, amt)
            if amt > 0:
                fill = exchange.create_market_sell_order(symbol, amt)
                try:
                    insert_order(symbol, "sell", float(amt), float(fill.get("price", 0.0)), info=fill)
                except Exception:
                    pass
                log_trade(symbol=symbol, side="sell", amount=float(amt), price=float(fill.get("price", 0.0)))
        except Exception as e:
            log(f"[brackets/close] sell error {symbol}: {e}")
            return
        br.amount_open = 0.0
        self.brackets.pop(symbol, None)
        try:
            send_telegram(f"🏁 CLOSE {symbol} ({reason}) amt={amt}")
        except Exception:
            pass

    def partial_close(self, exchange, symbol: str, fraction: float, reason: str) -> None:
        br = self.brackets.get(symbol)
        if not br or br.amount_open <= 0:
            return
        close_amt = max(0.0, min(br.amount_open, br.amount_total * fraction))
        if close_amt <= 0:
            return
        try:
            close_amt = self._guard_amount(exchange, symbol, close_amt)
            if close_amt > 0:
                fill = exchange.create_market_sell_order(symbol, close_amt)
                try:
                    insert_order(symbol, "sell", float(close_amt), float(fill.get("price", 0.0)), info=fill)
                except Exception:
                    pass
                br.amount_open = max(0.0, br.amount_open - close_amt)
                log_trade(symbol=symbol, side="sell", amount=float(close_amt), price=float(fill.get("price", 0.0)))
                try:
                    send_telegram(f"➗ PARTIAL {symbol} {fraction*100:.0f}% ({reason}) amt={close_amt}")
                except Exception:
                    pass
        except Exception as e:
            log(f"[partial_close] {symbol} error: {e}")

    # -------------------- orders ---------------------
    def market_buy(self, exchange, symbol: str, amount: float,
                   tp: Optional[float] = None, sl: Optional[float] = None,
                   trail_pct: Optional[float] = None, partial_pct: Optional[float] = None,
                   tp1_fraction: Optional[float] = None) -> Dict[str, Any]:
        amt = self._guard_amount(exchange, symbol, amount)
        if amt <= 0:
            raise ValueError("amount <= 0 after guards")

        if getattr(exchange, "create_market_buy_order", None) is None:
            raise RuntimeError("exchange lacks create_market_buy_order")

        fill = exchange.create_market_buy_order(symbol, amt)
        price = float(fill.get("price", 0.0))
        try:
            insert_order(symbol, "buy", float(amt), price, info=fill)
        except Exception:
            pass
        try:
            send_telegram(f"✅ BUY {symbol} amt={amt} @ {price}")
        except Exception:
            pass
        log_trade(symbol=symbol, side="buy", amount=float(amt), price=price)

        # open brackets/position
        if tp is not None and sl is not None and price > 0:
            self.open_position(symbol, price, amt, tp=float(tp), sl=float(sl),
                               trail_pct=trail_pct, partial_pct=partial_pct, tp1_fraction=tp1_fraction)
        return fill

    def market_sell(self, exchange, symbol: str, amount: float) -> Dict[str, Any]:
        amt = self._guard_amount(exchange, symbol, amount)
        if amt <= 0:
            raise ValueError("amount <= 0 after guards")
        if getattr(exchange, "create_market_sell_order", None) is None:
            raise RuntimeError("exchange lacks create_market_sell_order")

        fill = exchange.create_market_sell_order(symbol, amt)
        try:
            insert_order(symbol, "sell", float(amt), float(fill.get("price", 0.0)), info=fill)
        except Exception:
            pass
        try:
            send_telegram(f"✅ SELL {symbol} amt={amt}")
        except Exception:
            pass
        log_trade(symbol=symbol, side="sell", amount=float(amt), price=float(fill.get("price", 0.0)))
        return fill

    # ------------------ evaluation loop ---------------------
    def on_tick(self, exchange, symbols: Optional[list[str]] = None) -> None:
        """Call this regularly (e.g., each step) to enforce TP/SL/trailing/partials."""
        if not self.brackets:
            return
        to_check = list(self.brackets.keys()) if symbols is None else [s for s in symbols if s in self.brackets]
        for sym in to_check:
            br = self.brackets.get(sym)
            if not br or br.amount_open <= 0:
                continue
            # get current price
            try:
                t = exchange.fetch_ticker(sym)
                px = float(t.get("last") or t.get("close") or 0.0)
            except Exception as e:
                log(f"[tick] {sym} price error: {e}")
                continue
            if px <= 0:
                continue

            # update trailing high
            br.highest = max(br.highest, px)

            # thresholds
            tp_price1 = br.entry * (1.0 + br.tp * br.tp1_fraction)
            tp_price_full = br.entry * (1.0 + br.tp)
            sl_price = br.entry * (1.0 - br.sl)
            trail_stop = br.highest * (1.0 - br.trail_pct)

            # Partial TP at TP1
            if (not br.tp1_done) and px >= tp_price1 and br.partial_pct > 0:
                self.partial_close(exchange, sym, br.partial_pct, reason="TP1")
                br.tp1_done = True
                # after partial, tighten trailing (optional): halve trailing
                br.trail_pct = max(0.003, br.trail_pct * 0.5)

            # Full TP
            if px >= tp_price_full and br.amount_open > 0:
                self.close_position(exchange, sym, reason="TP")

            # Hard SL
            elif px <= sl_price and br.amount_open > 0:
                self.close_position(exchange, sym, reason="SL")

            # Trailing stop (only after some profit)
            elif br.amount_open > 0 and px <= trail_stop and br.highest > br.entry * 1.002:
                self.close_position(exchange, sym, reason="TRAIL")
