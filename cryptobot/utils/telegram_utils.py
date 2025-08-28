def format_trade_summary(
    symbol: str,
    entry_price: float,
    tp_price: float,
    sl_price: float,
    confidence: float,
    rel_pos: float,
    sentiment: str,
) -> str:
    """Format a human‑readable summary of a trade."""
    return (
        f"📈 Сделка: {symbol}\n"
        f"• Вход: {entry_price:.4f} | TP: {tp_price:.4f} | SL: {sl_price:.4f}\n"
        f"• Увереност: {confidence:.1%} | Позиция: {rel_pos:.1%}\n"
        f"• Настроение: {sentiment}"
    )

def format_status_summary(positions: dict) -> str:
    """Return a summary of all open positions."""
    if not positions:
        return "📊 Няма отворени позиции."
    summary = "📊 Активни позиции:\n"
    for s, d in positions.items():
        summary += (
            f"— {s.split('/')[0]} @ {d['entry']:.4f} | TP: {d['tp']:.4f}, SL: {d['sl']:.4f}\n"
        )
    return summary