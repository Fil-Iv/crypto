"""
Entry point for the fully automated trading agent.

This script orchestrates the entire workflow:

1. It runs a quick simulation to optimise the confidence threshold using
   recent market data. The selected threshold is stored in
   ``best_params.json``.
2. It instantiates the main trading bot and trains models for each
   symbol.
3. It enters a continuous loop where it executes trades and monitors
   positions. Every 6 hours the bot re-runs the simulation and
   retrains models to adapt to changing market conditions.

The goal is to provide a self-contained, self-improving agent that
requires no manual intervention beyond launching this script.

NOTE: In environments without ``ccxt`` or ``tensorflow``, the bot
falls back to using a dummy exchange and scikit-learn models via
imports in the underlying modules. Real trading will only happen when
valid API keys are supplied and the ``ccxt`` package is installed.
"""

import time
from datetime import datetime, timedelta, timezone

# bootstrp for direct run
import sys, pathlib
if __package__ in (None, "",):
    sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from utils.logger import log

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

from parameter_optimizer import optimise_parameters, update_best_params
from main_bot import SmartCryptoBot


def run_autobot():
    """Run the fully automated bot with periodic optimisation and training."""
    load_dotenv()
    # Step 1: run simulation to choose a threshold
    try:
        # Use symbols from config.json via SmartCryptoBot initialisation
        temp_bot = SmartCryptoBot()
        symbols = temp_bot.symbols
        del temp_bot
    except Exception:
        symbols = ["BTC/USDC", "ETH/USDC"]

    # Run initial optimisation with error handling
    try:
        sim_result = optimise_parameters(symbols)
        update_best_params(sim_result["threshold"])
        log(f"🎯 Оптимизиран праг: {sim_result['threshold']:.2f}")
    except Exception as e:
        log(f"[WARN] Оптимизацията пропадна: {e}. Ползвам текущия/дефолтен праг.")

    # Step 2: instantiate the trading bot
    bot = SmartCryptoBot()

    # Train models once before entering the trading loop
    bot.train_models()

    # Track last optimisation time
    last_optimisation = datetime.now(timezone.utc)

    while True:
        now = datetime.now(timezone.utc)

        # Re-run simulation and training every 6 hours
        if (now - last_optimisation).total_seconds() >= 6 * 3600:
            try:
                sim_result = optimise_parameters(bot.symbols)
                update_best_params(sim_result["threshold"])
                log(f"🔄 Нов праг от оптимизация: {sim_result['threshold']:.2f}")
                bot.threshold = sim_result["threshold"]
            except Exception as e:
                log(f"[WARN] Ре-оптимизацията пропадна: {e}. Запазвам стария праг.")
            bot.train_models()
            last_optimisation = now

        # Perform trading cycle
        bot.trade()
        time.sleep(300)


if __name__ == "__main__":
    run_autobot()