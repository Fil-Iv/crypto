# model_with_context.py
from __future__ import annotations

# --- bootstrap for script run (python auto_bot.py) ---
import sys, pathlib
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

# sklearn pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# absolute imports from project
from utils.data_cache import OHLCVCache
from utils.get_top_pairs import get_top_usdc_pairs
from utils.exchange_factory import get_exchange
from utils.add_indicators import add_indicators
# импорт на get_feature_columns от правилния модул utils
from utils.feature_config import get_feature_columns
from utils.logger import log  # ако твоят logger е в utils/

try:
    import joblib  # type: ignore
except Exception:
    joblib = None

MODELS_DIR = os.getenv("MODELS_DIR", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

@dataclass
class TrainConfig:
    timeframe: str = os.getenv("TRAIN_TIMEFRAME", "15m")
    limit: int = int(os.getenv("TRAIN_LIMIT", "1500"))
    symbols: int = int(os.getenv("TRAIN_SYMBOLS", "6"))
    test_frac: float = float(os.getenv("TRAIN_TEST_FRAC", "0.2"))

class ModelWithContext:
    """LogReg върху техн. фийчъри. predict_proba връща P(up) за последния ред."""
    def __init__(self, name: str = "baseline"):
        self.name = name
        self.path = os.path.join(MODELS_DIR, f"{self.name}.joblib")
        self.pipeline: Optional[Pipeline] = None
        self._feature_cols: List[str] = get_feature_columns()

    # --------------------- Training ---------------------
    def _build_dataset(self) -> pd.DataFrame:
        ex = get_exchange()
        cache = OHLCVCache()
        symbols = get_top_usdc_pairs(ex, limit=TrainConfig.symbols)

        frames = []
        for sym in symbols:
            try:
                df = cache.fetch(ex, sym, TrainConfig.timeframe, TrainConfig.limit)
                df = add_indicators(df)
                if df.empty or len(df) < 200:
                    continue
                df = df.copy()
                df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
                df["symbol"] = sym
                frames.append(df)
            except Exception as e:
                log(f"[model] dataset fetch {sym} error: {e}")
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=0).dropna()

    def _train(self, data: pd.DataFrame) -> None:
        X = data[self._feature_cols].values
        y = data["target"].values.astype(int)

        n = len(data)
        cut = int(n * (1.0 - TrainConfig.test_frac))
        X_train, X_test = X[:cut], X[cut:]
        y_train, y_test = y[:cut], y[cut:]

        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=None)),
        ])
        pipe.fit(X_train, y_train)
        self.pipeline = pipe

        try:
            if len(X_test) and len(np.unique(y_test)) > 1:
                p = pipe.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, p)
                log(f"[model] {self.name} trained. Test AUC={auc:.4f}")
        except Exception:
            pass

        if joblib is not None:
            try:
                joblib.dump({"pipeline": pipe, "features": self._feature_cols}, self.path)
            except Exception as e:
                log(f"[model] save error: {e}")

    def fit_if_needed(self) -> None:
        if self.pipeline is not None:
            return
        if joblib is not None and os.path.exists(self.path):
            try:
                obj = joblib.load(self.path)
                self.pipeline = obj["pipeline"]
                self._feature_cols = obj.get("features", self._feature_cols)
                return
            except Exception as e:
                log(f"[model] load failed, retraining: {e}")

        data = self._build_dataset()
        if data.empty:
            # fallback инициализация
            self.pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
            X = np.zeros((10, len(self._feature_cols)))
            y = np.array([0, 1] * 5)
            self.pipeline.fit(X, y)
            log("[model] fallback initialized (no data).")
            return
        self._train(data)

    # --------------------- Inference ---------------------
    def predict_proba(self, feat_df: pd.DataFrame) -> float:
        if self.pipeline is None:
            self.fit_if_needed()
        if feat_df is None or feat_df.empty:
            return 0.5
        last = feat_df[self._feature_cols].tail(1).values
        try:
            p = float(self.pipeline.predict_proba(last)[0, 1])
            return max(0.0, min(1.0, p))
        except Exception as e:
            log(f"[model] predict error: {e}")
            return 0.5
