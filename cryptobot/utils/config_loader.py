# config_loader.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any

def _load_dotenv():
    try:
        from dotenv import load_dotenv  # pip install python-dotenv
        # търси .env в корена на проекта
        root = Path(__file__).resolve().parent
        env_path = root / ".env"
        load_dotenv(dotenv_path=env_path)  # не гърми ако липсва
    except Exception:
        pass

_load_dotenv()

def load_api_credentials() -> Dict[str, Any]:
    """Връща API ключовете за борсата от .env/ENV променливи."""
    return {
        "apiKey": os.getenv("BINANCE_API_KEY", "") or os.getenv("API_KEY", ""),
        "secret": os.getenv("BINANCE_API_SECRET", "") or os.getenv("API_SECRET", ""),
    }

def load_telegram() -> Dict[str, Any]:
    return {
        "enabled": str(os.getenv("TELEGRAM_ENABLED", "false")).lower() == "true",
        "token": os.getenv("TELEGRAM_TOKEN", ""),
        "chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    }

def load_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() in {"1","true","yes","y","on"}

def load_float(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except Exception: return default
