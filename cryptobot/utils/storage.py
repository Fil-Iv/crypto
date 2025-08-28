# storage.py
from __future__ import annotations
import os, sqlite3, json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict

DB_PATH = Path(os.getenv("BOT_DB_PATH", "data/bot.sqlite"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _conn():
    return sqlite3.connect(str(DB_PATH))

def init_db() -> None:
    con = _conn()
    cur = con.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS bars(
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY(ts, symbol, timeframe)
        );
        CREATE TABLE IF NOT EXISTS signals(
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            prob REAL,
            threshold REAL,
            params TEXT
        );
        CREATE TABLE IF NOT EXISTS orders(
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT,
            amount REAL,
            price REAL,
            info TEXT
        );
        CREATE TABLE IF NOT EXISTS fills(
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT,
            amount REAL,
            price REAL,
            fee REAL,
            info TEXT
        );
        CREATE TABLE IF NOT EXISTS pnl(
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            pnl REAL,
            note TEXT
        );
        CREATE TABLE IF NOT EXISTS metrics(
            ts TEXT NOT NULL,
            key TEXT NOT NULL,
            value REAL
        );
        """
    )
    con.commit()
    con.close()

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

def insert_signal(symbol: str, prob: float, threshold: float, params: Dict[str, Any]) -> None:
    con = _conn(); cur = con.cursor()
    cur.execute("INSERT INTO signals VALUES (?,?,?, ?,?)",
                (_utcnow(), symbol, float(prob), float(threshold), json.dumps(params)))
    con.commit(); con.close()

def insert_order(symbol: str, side: str, amount: float, price: float, info: Dict[str, Any] | None = None) -> None:
    con = _conn(); cur = con.cursor()
    cur.execute("INSERT INTO orders VALUES (?,?,?,?,?,?)",
                (_utcnow(), symbol, side, float(amount), float(price), json.dumps(info or {})))
    con.commit(); con.close()

def insert_fill(symbol: str, side: str, amount: float, price: float, fee: float = 0.0, info: Dict[str, Any] | None = None) -> None:
    con = _conn(); cur = con.cursor()
    cur.execute("INSERT INTO fills VALUES (?,?,?,?,?,?)",
                (_utcnow(), symbol, side, float(amount), float(price), float(fee)))
    con.commit(); con.close()

def insert_metric(key: str, value: float) -> None:
    con = _conn(); cur = con.cursor()
    cur.execute("INSERT INTO metrics VALUES (?,?,?)", (_utcnow(), key, float(value)))
    con.commit(); con.close()
