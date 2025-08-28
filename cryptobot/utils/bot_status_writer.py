import os

def mark_bot_running() -> None:
    """Create a flag file indicating the bot is running."""
    try:
        with open("bot_running.flag", "w", encoding="utf-8") as f:
            f.write("running")
    except Exception:
        # Silently ignore any errors writing the flag
        pass

def clear_flag() -> None:
    """Remove the running flag file if it exists."""
    try:
        if os.path.exists("bot_running.flag"):
            os.remove("bot_running.flag")
    except Exception:
        # Ignore errors removing the flag
        pass