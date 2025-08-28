"""Utility package for SmartCryptoBot.

This package contains helper modules used by the main trading bot and backtest
scripts. Where possible, these implementations are intentionally simple and
conservative. They avoid any external network calls and instead fall back
to sensible defaults so the bot can operate in a self‑contained manner.

The goal is to provide enough functionality to allow the trading bot to
initialise, train, and simulate trades without raising import errors. Feel
free to extend these modules to suit your own needs.
"""
