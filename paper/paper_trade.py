"""
Paper trading en Alpaca con la estrategia de strategy.py.

Uso:
  python paper_trade.py            # solo opera si hoy es el primer día hábil del mes
  python paper_trade.py --force    # rebalancea hoy pase lo que pase (para la primera prueba)
  python paper_trade.py --dry-run  # calcula y muestra las órdenes sin enviarlas

Variables de entorno (archivo .env o secrets de GitHub):
  ALPACA_API_KEY, ALPACA_SECRET_KEY        (cuenta PAPER de Alpaca)
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID     (opcional: aviso por Telegram)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from strategy import UNIVERSE, LOOKBACK_DAYS, target_weights, momentum_scores

load_dotenv(Path(__file__).parent / ".env")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------

def alpaca_clients():
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    key, sec = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    return TradingClient(key, sec, paper=True), StockHistoricalDataClient(key, sec)


def fetch_prices(data_client, symbols, days=LOOKBACK_DAYS + 40) -> pd.DataFrame:
    """Cierres diarios ajustados hasta AYER (nunca incluye el día de hoy)."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
                           start=end - timedelta(days=int(days * 1.6)), end=end,
                           adjustment=Adjustment.ALL, feed=DataFeed.IEX)
    bars = data_client.get_stock_bars(req).df
    prices = bars["close"].unstack(level=0)
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    return prices.dropna()


def is_first_trading_day_of_month(trading_client) -> bool:
    from alpaca.trading.requests import GetCalendarRequest
    today = datetime.now(timezone.utc).date()
    cal = trading_client.get_calendar(GetCalendarRequest(start=today.replace(day=1), end=today))
    open_days = [c.date for c in cal]
    return bool(open_days) and open_days[0] == today


# ---------------------------------------------------------------------------
# Rebalanceo
# ---------------------------------------------------------------------------

def plan_orders(trading_client, targets: pd.Series) -> list[dict]:
    """Compara pesos objetivo con posiciones actuales y devuelve órdenes en $ (notional)."""
    acct = trading_client.get_account()
    equity = float(acct.equity)
    current = {p.symbol: float(p.market_value) for p in trading_client.get_all_positions()}
    orders = []
    for sym in sorted(set(targets.index) | set(current)):
        target_val = equity * float(targets.get(sym, 0.0))
        diff = target_val - current.get(sym, 0.0)
        if abs(diff) < max(25.0, 0.002 * equity):   # ignora ajustes minúsculos
            continue
        orders.append({"symbol": sym, "side": "buy" if diff > 0 else "sell", "notional": round(abs(diff), 2)})
    # vender primero para liberar efectivo
    orders.sort(key=lambda o: o["side"] != "sell")
    return orders


def submit(trading_client, orders: list[dict]) -> list[dict]:
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    results = []
    for o in orders:
        req = MarketOrderRequest(symbol=o["symbol"], notional=o["notional"],
                                 side=OrderSide.BUY if o["side"] == "buy" else OrderSide.SELL,
                                 time_in_force=TimeInForce.DAY)
        resp = trading_client.submit_order(req)
        results.append({**o, "order_id": str(resp.id), "status": str(resp.status)})
    return results


# ---------------------------------------------------------------------------
# Registro y aviso
# ---------------------------------------------------------------------------

def notify(text: str):
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        return
    try:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      json={"chat_id": chat, "text": text}, timeout=10)
    except Exception as e:  # el aviso nunca debe tumbar el rebalanceo
        print("telegram error:", e)


def log(record: dict):
    path = LOG_DIR / f"{datetime.now(timezone.utc):%Y-%m-%d}.json"
    path.write_text(json.dumps(record, indent=2, default=str))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    trading, data = alpaca_clients()

    if not args.force and not is_first_trading_day_of_month(trading):
        print("Hoy no es el primer día hábil del mes. Nada que hacer.")
        return

    prices = fetch_prices(data, UNIVERSE)
    scores = momentum_scores(prices)
    targets = target_weights(prices)
    orders = plan_orders(trading, targets)
    acct = trading.get_account()

    lines = [f"Momentum ETF — rebalanceo {datetime.now(timezone.utc):%Y-%m-%d}",
             f"Equity: ${float(acct.equity):,.0f}",
             "Scores 6m: " + ", ".join(f"{s} {v:+.1%}" for s, v in scores.sort_values(ascending=False).items()),
             "Objetivo: " + (", ".join(f"{s} {w:.0%}" for s, w in targets[targets > 0].items()) or "100% efectivo"),
             "Órdenes: " + (", ".join(f"{o['side']} {o['symbol']} ${o['notional']:,.0f}" for o in orders) or "ninguna")]
    msg = "\n".join(lines)
    print(msg)

    results = [] if args.dry_run else submit(trading, orders)
    if args.dry_run:
        print("(dry-run: no se envió nada)")

    log({"date": str(datetime.now(timezone.utc)), "equity": float(acct.equity), "last_price_date": str(prices.index[-1].date()),
         "scores": scores.round(4).to_dict(), "targets": targets.round(4).to_dict(), "orders": results or orders,
         "dry_run": args.dry_run})
    if not args.dry_run:
        notify(msg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        notify(f"Momentum ETF — ERROR: {e}")
        raise
