import os
import pandas as pd
from .logger import log

def log_trade(symbol: str, trade_data: dict) -> None:
    """Append a trade record to the symbol's history CSV file.

    The data is stored in ``history/<symbol>_trades.csv`` where ``symbol``
    has its slash replaced by an underscore. If the file already exists
    the new record is concatenated to the existing data frame. On any
    error the exception is suppressed and logged.

    :param symbol: Trading pair symbol such as ``BTC/USDC``.
    :param trade_data: A dictionary of data to record.
    """
    os.makedirs("history", exist_ok=True)
    filepath = f"history/{symbol.replace('/', '_')}_trades.csv"
    df = pd.DataFrame([trade_data])
    if os.path.exists(filepath):
        try:
            old = pd.read_csv(filepath)
        except pd.errors.EmptyDataError:
            old = pd.DataFrame()
        combined = pd.concat([old, df], ignore_index=True)
    else:
        combined = df
    try:
        combined.to_csv(filepath, index=False)
        log(f"[log_trade] Записана сделка за {symbol}")
    except Exception:
        log(f"[log_trade] Failed to write trade for {symbol}")