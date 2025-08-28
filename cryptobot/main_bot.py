import os
import time
import json
import numpy as np
import pandas as pd
# Attempt to import TensorFlow for compatibility. If unavailable the
# variable ``tf`` will be ``None`` and TensorFlow models will not be
# used. All model creation is handled via ``utils.model_versions``.
try:
    import tensorflow as tf  # type: ignore
except Exception:
    tf = None  # type: ignore

import joblib
from typing import Any, Dict, Optional
from sklearn.linear_model import LogisticRegression

# -----------------------------------------------------------------------------
# Model builder
# -----------------------------------------------------------------------------
# utils.model_versions.py in този проект няма build_model функция. Тук
# дефинираме прост генератор на модели на база посочения тип. Ако бъде
# подаден тип "lstm", се изгражда LSTM слоеве; иначе се използва
# базов плътен (Dense) модел. Всички модели се компилират с Adam
# оптимизатор и binary_crossentropy.

def build_model(model_type: str, input_shape: tuple[int, int]) -> Any:
    """Construct a TensorFlow/Keras model depending on the model_type.

    Args:
        model_type: Name of the model architecture (e.g. "lstm" or "baseline").
        input_shape: Shape of the input data (timesteps, features).

    Returns:
        A compiled Keras model.
    """
    # If TensorFlow is not available, fall back to a simple scikit-learn model
    # but here we assume tf is available from the earlier import. If not,
    # fallback to a dummy model which returns 0.5 probability.
    if tf is None:
        class DummyModel:
            def predict(self, X, verbose=0):
                return np.full((len(X), 1), 0.5)

            def fit(self, X, y, epochs=1, batch_size=32, verbose=0, validation_split=0.0):
                return self

            def save(self, path):
                pass

        return DummyModel()

    model_type = (model_type or "baseline").lower()
    if model_type == "lstm":
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=input_shape),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
    else:
        # Baseline dense model
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import MinMaxScaler
# Attempt to import load_dotenv from python-dotenv. If unavailable, fall back
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):
        """Fallback no-op when python-dotenv is not installed."""
        return None

from utils.exchange_factory import get_exchange, DRY_RUN
from utils.bot_status_writer import mark_bot_running, clear_flag
from utils.add_indicators import add_indicators
from utils.logger import log
from utils.notifier import send_telegram
# build_model е имплементиран локално по-долу. Не използваме utils.model_versions.build_model.
from utils.model_selector import evaluate_models
from utils.get_top_pairs import get_top_usdc_pairs
from utils.trade_manager import execute_trade, monitor_positions, send_periodic_report
from utils.capital_manager import load_capital_state, get_available_capital
from utils.feature_config import FEATURE_COLUMNS
from utils.telegram_utils import format_trade_summary, format_status_summary
from utils.adaptive_strategy import adaptive_parameters
from utils.dummy_exchange import DummyExchange  # оставен за съвместимост, дори да не се ползва
from utils.metrics_server import start_metrics_server

PARAMS_PATH = "best_params.json"

class SmartCryptoBot:
    def __init__(self):
        load_dotenv()

        # Централизиран exchange (singleton). Ако няма ключове → DRY_RUN=True.
        self.exchange = get_exchange()
        if DRY_RUN:
            log("ℹ️ DRY_RUN=True – private endpoints (balance/orders) се пропускат, няма заредени ключове.")

        # Списък с търгувани символи (най‑ликвидните USDC двойки).
        self.symbols = get_top_usdc_pairs(self.exchange, limit=20)
        # Кеш от обучени модели за всяка валутна двойка
        self.model_map: Dict[str, Any] = {}
        # Кеш от скалери за нормализиране на характеристиките
        self.scaler_map: Dict[str, MinMaxScaler] = {}
        # Кеш от калибратори на вероятностите (логистична регресия) за всяка валутна двойка
        self.calibrator_map: Dict[str, Any] = {}
        # Зареждаме предходните позиции от файл
        self.positions = self.load_positions()

        now = datetime.now(timezone.utc)
        self.last_train = now
        self.last_model_selection = now

        selection = evaluate_models()
        adapt = adaptive_parameters()

        self.selected_model_name = selection["model"]
        # Базов праг на увереност от адаптивния модул; ще бъде коригиран след обучение и новини
        self.threshold = adapt["threshold"]
        self.tp = adapt["tp"]
        self.sl = adapt["sl"]
        self.aggressiveness = float(adapt["aggressiveness"])

        send_telegram(f"🤖 Ботът стартира. Модел: {self.selected_model_name.upper()}")

        # Зареди конфигурация за анализ на новини (ако съществува)
        self.config = self._load_config()
        # Настрой теглото на новините (колко силно влияят върху прага). Стойност 0.05 намалява или увеличава прага до ±5 %.
        self.sentiment_weight: float = float(self.config.get("sentiment_weight", 0.05))


    def load_positions(self):
        try:
            with open("positions.json", "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_positions(self):
        try:
            with open("positions.json", "w") as f:
                json.dump(self.positions, f, indent=2)
        except Exception as e:
            log(f"[save_positions] Error: {e}")

    # ------------------------------------------------------------------
    # Конфигурация и анализ на новини
    # ------------------------------------------------------------------
    def _load_config(self) -> Dict[str, Any]:
        """Load optional runtime configuration from config.json.

        The configuration file may contain API keys, trading parameters
        and a list of recent news headlines under the key ``latest_news``.
        A sentiment weight under ``sentiment_weight`` can control the
        impact of news on the trading threshold. Returns an empty dict
        when the file does not exist or cannot be parsed.
        """
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _get_news_sentiment(self) -> float:
        """Calculate a simple sentiment score in the range [-1, 1] based on recent news.

        The method scans the headlines stored under ``latest_news`` in
        the configuration file. It counts occurrences of positive and
        negative keywords and normalizes the difference by the number of
        headlines. A positive score implies bullish sentiment (optimism),
        leading to a slightly lower confidence threshold, whereas a
        negative score does the opposite. The chosen words are broad and
        can be adjusted in the configuration file if needed.
        """
        cfg = self.config or {}
        headlines = cfg.get("latest_news", []) or []
        if not isinstance(headlines, list):
            return 0.0
        # Define a small set of keywords for sentiment analysis
        positive_words = ["rise", "bull", "up", "growth", "improve", "increase", "record", "breakout", "adoption", "scale"]
        negative_words = ["fall", "bear", "down", "decline", "decrease", "drop", "loss", "regulation", "risk", "ban"]

        score = 0
        for h in headlines:
            if not isinstance(h, str):
                continue
            text = h.lower()
            pos_count = sum(text.count(w) for w in positive_words)
            neg_count = sum(text.count(w) for w in negative_words)
            score += pos_count - neg_count

        # Normalize: divide by max possible counts to keep within [-1, 1]
        max_possible = len(headlines) * max(1, len(positive_words))
        if max_possible == 0:
            return 0.0
        normalized = max(-1.0, min(1.0, score / max_possible))
        return normalized

    def fetch_data(self, symbol, limit=12000):
        def fetch_and_process(tf):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                return add_indicators(df)
            except Exception as e:
                log(f"Error fetching {tf} data for {symbol}: {e}")
                return pd.DataFrame()

        return (
            fetch_and_process("15m"),
            fetch_and_process("1h"),
            fetch_and_process("4h")
        )

    def prepare_data(self, df_10m, df_1h, df_4h):
        try:
            df = df_10m.copy().dropna()
            df["MACD_1h"] = df_1h["MACD"].reindex(df.index, method="nearest").bfill()
            df["RSI_4h"] = df_4h["RSI_14"].reindex(df.index, method="nearest").bfill()

            features = df[FEATURE_COLUMNS].dropna().values
            if len(features) < 31:
                return None, None, None

            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(features)

            X, y = [], []
            for i in range(30, len(scaled) - 1):
                X.append(scaled[i - 30:i])
                label = 1 if scaled[i + 1][0] > scaled[i][0] else 0
                y.append(label)

            return np.array(X), np.array(y), scaler
        except Exception as e:
            log(f"Error in prepare_data: {e}")
            return None, None, None

    def train_models(self):
        """Train a predictive model for each symbol.

        In addition to fitting the neural network, this method also trains a
        probability calibrator (logistic regression) on the raw model
        predictions. The calibrator learns to map the network's outputs to
        calibrated probabilities and provides a data‑driven confidence
        threshold. After training all calibrators the median of their 0.5
        thresholds replaces the existing `self.threshold`. This dynamic
        adjustment helps the bot remain sensitive to current market
        conditions without overfitting to a single instrument.
        """
        threshold_candidates: list[float] = []
        for symbol in self.symbols:
            try:
                # Fetch training data for multiple timeframes
                df_10m, df_1h, df_4h = self.fetch_data(symbol)
                X, y, scaler = self.prepare_data(df_10m, df_1h, df_4h)
                # Skip if not enough data
                if X is None or len(X) == 0:
                    continue

                # Build and fit the primary model
                model = build_model(model_type=self.selected_model_name,
                                    input_shape=(30, len(FEATURE_COLUMNS)))
                model.fit(X, y, epochs=5, batch_size=32, verbose=0, validation_split=0.1)

                # Cache model and scaler
                self.model_map[symbol] = model
                self.scaler_map[symbol] = scaler

                # Save artifacts to disk
                try:
                    model.save(f"models/{symbol.replace('/', '_')}_model.keras")
                    joblib.dump(scaler, f"models/{symbol.replace('/', '_')}_scaler.pkl")
                except Exception as e:
                    log(f"[train_models] saving artifacts for {symbol} failed: {e}")

                # Train a logistic regression calibrator using model's raw predictions
                try:
                    raw_preds = model.predict(X, verbose=0).reshape(-1)
                    # Logistic regression expects 2 classes; skip if only one class present
                    if len(set(y)) > 1:
                        cal = LogisticRegression()
                        cal.fit(raw_preds.reshape(-1, 1), y)
                        self.calibrator_map[symbol] = cal
                        # Derive the prediction threshold at which P=0.5: intercept + coef*x = 0
                        coef = float(cal.coef_[0][0])
                        intercept = float(cal.intercept_[0])
                        if abs(coef) > 1e-6:
                            thr = -intercept / coef
                            threshold_candidates.append(float(thr))
                except Exception as e:
                    log(f"[train_models] calibrator error for {symbol}: {e}")
            except Exception as e:
                log(f"Train error {symbol}: {e}")

        # Update global threshold based on median of calibrators' thresholds
        if threshold_candidates:
            try:
                median_thr = float(np.median(threshold_candidates))
                # Constrain within ±0.1 of current threshold to avoid huge jumps
                self.threshold = float(max(self.threshold - 0.1, min(self.threshold + 0.1, median_thr)))
                log(f"[train_models] recalibrated threshold to {self.threshold:.4f}")
            except Exception as e:
                log(f"[train_models] threshold recalibration error: {e}")

    def predict(self, symbol, df_10m, df_1h, df_4h):
        """Return calibrated probability of upward movement for the given symbol.

        The raw network output is optionally passed through a logistic
        regression calibrator trained during ``train_models``. If no model
        or calibrator is available, returns 0.5 (neutral confidence).
        """
        model = self.model_map.get(symbol)
        scaler = self.scaler_map.get(symbol)
        if model is None or scaler is None:
            return 0.5

        try:
            df = df_10m.copy().dropna()
            # Align higher timeframe indicators with 10m data
            df["MACD_1h"] = df_1h["MACD"].reindex(df.index, method="nearest").bfill()
            df["RSI_4h"] = df_4h["RSI_14"].reindex(df.index, method="nearest").bfill()

            features = df[FEATURE_COLUMNS].dropna().values
            if len(features) < 31:
                return 0.5

            scaled = scaler.transform(features)
            # Feed last 30 timesteps into the model
            X = np.expand_dims(scaled[-30:], axis=0)
            raw_pred = float(model.predict(X, verbose=0)[0][0])

            # Calibrate raw prediction if a calibrator exists
            calibrator = self.calibrator_map.get(symbol)
            if calibrator is not None:
                # logistic regression returns probabilities of classes [0, 1], select positive class probability
                try:
                    calibrated = float(calibrator.predict_proba(np.array([[raw_pred]]))[0][1])
                    return calibrated
                except Exception:
                    # fallback to raw prediction on error
                    return raw_pred
            # fallback: return raw network output
            return raw_pred
        except Exception as e:
            log(f"Prediction error for {symbol}: {e}")
            return 0.5

    def trade(self):
        predictions: Dict[str, tuple[pd.DataFrame, float]] = {}
        new_trades: list[str] = []
        # Iterate through symbols and compute calibrated predictions
        for symbol in self.symbols:
            try:
                df_10m, df_1h, df_4h = self.fetch_data(symbol)
                conf = self.predict(symbol, df_10m, df_1h, df_4h)
                predictions[symbol] = (df_10m, conf)
                log(f"{symbol} prediction: {conf:.4f}")
            except Exception as e:
                log(f"Prediction error: {e}")

        # Compute a sentiment‑adjusted threshold once per trading cycle
        sentiment = self._get_news_sentiment()
        # Lower the threshold when news are positive (encouraging more trades), increase when negative
        adjusted_threshold = self.threshold * (1 - self.sentiment_weight * sentiment)

        capital_state = load_capital_state()

        for symbol, (df, conf) in predictions.items():
            # Only open a new position if we don't already have one
            if symbol not in self.positions:
                cap = get_available_capital(capital_state, symbol)
                execute_trade(
                    symbol,
                    df,
                    conf,
                    self.exchange,
                    cap,
                    self.positions,
                    threshold=adjusted_threshold,
                    tp=self.tp,
                    sl=self.sl,
                    aggressiveness=self.aggressiveness,
                )
                # Persist new positions to disk
                self.save_positions()

                pos = self.positions.get(symbol)
                if pos:
                    summary = format_trade_summary(
                        symbol=symbol,
                        entry_price=pos.get("entry", 0),
                        tp_price=pos.get("tp", 0),
                        sl_price=pos.get("sl", 0),
                        confidence=conf,
                        rel_pos=pos.get("relative_position", 0),
                        sentiment=pos.get("sentiment", "neutral"),
                    )
                    new_trades.append(summary)

        # Monitor open positions
        monitor_positions(self.exchange, self.positions, None)

        if new_trades:
            send_telegram("📈 Нови сделки:\n\n" + "\n\n".join(new_trades))

    def load_best_params(self):
        try:
            with open(PARAMS_PATH, "r", encoding="utf-8")                as f:
                return json.load(f)
        except Exception:
            return None

if __name__ == "__main__":
    try:
        mark_bot_running()
        # start metrics server
        try:
            start_metrics_server()
        except Exception:
            pass
        bot = SmartCryptoBot()

        while True:
            now = datetime.now(timezone.utc)

            if (now - bot.last_train).total_seconds() > 6 * 3600:
                selection = evaluate_models()
                adapt = adaptive_parameters()
                bot.selected_model_name = selection["model"]
                bot.threshold = adapt["threshold"]
                bot.tp = adapt["tp"]
                bot.sl = adapt["sl"]
                bot.aggressiveness = adapt["aggressiveness"]

                send_telegram(f"🤖 Нов модел избран: {bot.selected_model_name}")
                bot.train_models()
                send_telegram("🧠 Моделите са обновени.")
                bot.last_train = now
                bot.last_model_selection = now

            bot.trade()
            # изпращаме периодичен отчет за натрупаните решения
            send_periodic_report()
            time.sleep(300)

    finally:
        clear_flag()
