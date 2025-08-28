import os
from datetime import datetime
import csv

def log(message: str) -> None:
    """Append a log message to ``log.txt`` and print it.

    This helper adds a UTC timestamp to each message and writes it to
    ``log.txt``. If any error occurs during file operations the
    exception is suppressed to avoid interrupting the caller. The
    message is always echoed to standard output.

    :param message: The text to log.
    """
    try:
        timestamp = datetime.utcnow().isoformat()
        log_line = f"[{timestamp}] {message}\n"
        with open("log.txt", "a", encoding="utf-8") as f:
            f.write(log_line)
        # Print without timestamp for readability
        print(message)
    except Exception:
        # Ignore logging errors silently
        pass

def log_trade(data: dict, file: str = "real_trades.csv") -> None:
    """Append a trade record to a CSV file.

    This convenience function writes a dictionary representing a trade
    to ``real_trades.csv`` (or another specified file). If the file
    does not exist the CSV header is written first.

    :param data: A dictionary with keys matching the headers defined
        below. Unexpected keys are ignored.
    :param file: Path to the CSV file. Defaults to ``real_trades.csv``.
    """
    headers = [
        "timestamp",
        "symbol",
        "side",
        "entry_price",
        "exit_price",
        "amount",
        "pnl_percent",
        "confidence",
        "rel_pos",
        "sentiment",
    ]
    file_exists = os.path.isfile(file)
    try:
        with open(file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            # Only write keys that are in headers
            filtered = {k: data.get(k, "") for k in headers}
            writer.writerow(filtered)
    except Exception:
        # Ignore file errors silently
        pass