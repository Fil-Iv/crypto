# feature_config.py
from __future__ import annotations

from typing import List

# Единствен източник на истина за фийчърите.
# Трябва да съвпадат с колоните, които създаваме в add_indicators.add_indicators().

_BASE_FEATURES: List[str] = [
    "ret_1",
    "ret_5",
    "ret_20",
    "vol_20",
    "atr_14",
    "rsi_14",
    "mom_10",
    "ema_20",
    "ema_50",
    "ema_diff",
    # лагове за избягване на leakage
    "ret_1_lag1",
    "ret_1_lag2",
    "ret_1_lag3",
    "rsi_14_lag1",
    "rsi_14_lag2",
    "rsi_14_lag3",
    "ema_diff_lag1",
    "ema_diff_lag2",
    "ema_diff_lag3",
]

def get_feature_columns() -> List[str]:
    # връщаме копие, за да не се модифицира глобалният списък
    return list(_BASE_FEATURES)
