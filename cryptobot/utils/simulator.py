# simulator.py
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from logger import log
from notifier import send_telegram

from __future__ import annotations
from ..logger import log
from ..notifier import send_telegram
import time

class PaperSimulator:
    '''Paper-trading runner that reuses live bot logic but prevents real orders.'''
    def __init__(self, bot_cls, interval_min: int = 15):
        self.bot = bot_cls()
        self.interval = interval_min * 60
        try:
            from .exchange_factory import DRY_RUN
            DRY_RUN = True
        except Exception:
            pass

    def run_forever(self):
        send_telegram("🧪 Стартиран paper-trade симулатор.")
        while True:
            try:
                out = self.bot.step()
                log(f"[SIM] {out}")
            except Exception as e:
                log(f"[SIM] loop error: {e}")
            time.sleep(self.interval)
