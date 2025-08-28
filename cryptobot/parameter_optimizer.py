# parameter_optimizer.py
# Random search за threshold/TP/SL върху Backtester; работи като скрипт и модул.

# --- bootstrap за директно пускане (python auto_bot.py) ---
import sys, pathlib
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))        # за utils/*
sys.path.insert(0, str(ROOT.parent)) # ако проектът е пакет

import os, json, random
from dataclasses import dataclass
from typing import Dict, List

from utils.exchange_factory import get_exchange
from utils.get_top_pairs import get_top_usdc_pairs
from utils.backtester import Backtester
from model_with_context import ModelWithContext
# изтегляме logger от utils, тъй като няма logger.py на най-горно ниво
from utils.logger import log

@dataclass
class SearchSpace:
    threshold_min: float = float(os.getenv("OPT_THR_MIN", "0.52"))
    threshold_max: float = float(os.getenv("OPT_THR_MAX", "0.70"))
    tp_min: float = float(os.getenv("OPT_TP_MIN", "0.007"))
    tp_max: float = float(os.getenv("OPT_TP_MAX", "0.05"))
    sl_min: float = float(os.getenv("OPT_SL_MIN", "0.004"))
    sl_max: float = float(os.getenv("OPT_SL_MAX", "0.04"))
    samples: int = int(os.getenv("OPT_SAMPLES", "40"))
    timeframe: str = os.getenv("OPT_TIMEFRAME", "15m")
    symbols_limit: int = int(os.getenv("OPT_SYMBOLS", "6"))

def _rand(a: float, b: float) -> float:
    return random.random() * (b - a) + a

def search_params() -> Dict[str, float]:
    sp = SearchSpace()
    random.seed(42)

    ex = get_exchange()
    symbols = get_top_usdc_pairs(ex, limit=sp.symbols_limit)
    log(f"[opt] symbols = {symbols}")

    model = ModelWithContext("baseline")
    try:
        model.fit_if_needed()
    except Exception:
        pass

    best = None  # (score, params)
    for i in range(sp.samples):
        params = {
            "threshold": round(_rand(sp.threshold_min, sp.threshold_max), 4),
            "tp": round(_rand(sp.tp_min, sp.tp_max), 4),
            "sl": round(_rand(sp.sl_min, sp.sl_max), 4),
        }
        bt = Backtester(ex, model, params, timeframe=sp.timeframe)
        res = bt.run(symbols, limit=1200)
        if not res:
            continue

        avg_sharpe = sum(r.sharpe for r in res) / max(1, len(res))
        avg_pnl = sum(r.pnl_pct for r in res) / max(1, len(res))
        trades = sum(r.trades for r in res)
        penalty = 0.0 if trades >= 20 else -0.2
        score = avg_sharpe + 0.5 * avg_pnl + penalty

        log(f"[opt] {i+1}/{sp.samples} params={params} -> score={score:.4f}, trades={trades}")
        if best is None or score > best[0]:
            best = (score, params)

    if best is None:
        best = (0.0, {"threshold": 0.58, "tp": 0.02, "sl": 0.01})
    log(f"[opt] BEST {best[1]} score={best[0]:.4f}")
    return best[1]

def write_best_params(path: str = "config/params.json") -> Dict[str, float]:
    params = search_params()
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return params

# legacy алиаси (както auto_bot.py очаква)
optimise_parameters = search_params
update_best_params = lambda thr: write_best_params()  # записва целия сет; ако ти трябва само threshold, чети/пипни JSON-а.

if __name__ == "__main__":
    out = write_best_params()
    print(json.dumps(out, indent=2))
