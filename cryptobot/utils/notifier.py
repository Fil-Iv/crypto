import os
import requests
# Attempt to import load_dotenv; if unavailable, define a no‑op
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):
        return None


def send_telegram(message: str) -> None:
    """Send a Telegram message if notifications are enabled.

    Reads the ``TELEGRAM_ENABLED``, ``TELEGRAM_TOKEN`` and
    ``TELEGRAM_CHAT_ID`` variables from the environment via
    ``python-dotenv``. If Telegram is disabled or credentials are
    missing the message is silently ignored. Any exceptions during
    transmission are caught and logged to standard output.

    :param message: The message text to send.
    """
    try:
        # Load environment variables from .env file if present
        load_dotenv()
        enabled = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
        if not enabled:
            return
        token = os.getenv("TELEGRAM_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        # Use ``requests.post`` to dispatch the message
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        # Print the error but do not raise
        print(f"[Telegram Error] {e}")