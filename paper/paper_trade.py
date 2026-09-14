"""
Paper trading en Alpaca. Una cuenta paper por estrategia.

Uso:
  python paper_trade.py --strategy MOM6            # opera solo el primer día hábil del mes
  python paper_trade.py --strategy VOLT --force    # rebalancea hoy (primera entrada)
  python paper_trade.py --strategy DUAL --dry-run  # muestra las órdenes sin enviarlas

Claves por estrategia (en .env o secrets de GitHub):
  MOM6:  ALPACA_API_KEY / ALPACA_SECRET_KEY            (las originales)
  VOLT:  ALPACA_API_KEY_VOLT / ALPACA_SECRET_KEY_VOLT
  DUAL:  ALPACA_API_KEY_DUAL / ALPACA_SECRET_KEY_DUAL
Opcional: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (compartidos).
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from strategy import STRATEGIES, UNIVERSE, symbols_needed, target_weights, momentum_scores

load_dotenv(Path(__file__).parent / ".env")
LOG_DIR = Path(__file__).parent / "logs"


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------

def credentials(name: str) -> tuple[str, str]:
    suffix = "" if name == "MOM6" else f"_{name}"
    try:
        return os.environ[f"ALPACA_API_KEY{suffix}"], os.environ[f"ALPACA_SECRET_KEY{suffix}"]
    except KeyError as e:
        raise SystemExit(f"Falta la variable {e} para la estrategia {name}")


def alpaca_clients(name: str):
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    key, sec = credentials(name)
    return TradingClient(key, sec, paper=True), StockHistoricalDataClient(key, sec)


def fetch_prices(data_client, symbols, days) -> pd.DataFrame:
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
    return prices[symbols].dropna()


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
    equity = float(trading_client.get_account().equity)
    current = {p.symbol: float(p.market_value) for p in trading_client.get_all_positions()}
    orders = []
    for sym in sorted(set(targets.index) | set(current)):
        diff = equity * float(targets.get(sym, 0.0)) - current.get(sym, 0.0)
        if abs(diff) < max(25.0, 0.002 * equity):
            continue
        orders.append({"symbol": sym, "side": "buy" if diff > 0 else "sell", "notional": round(abs(diff), 2)})
    orders.sort(key=lambda o: o["side"] != "sell")   # vender primero
    return orders


def submit(trading_client, orders: list[dict]) -> list[dict]:
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    out = []
    for o in orders:
        req = MarketOrderRequest(symbol=o["symbol"], notional=o["notional"],
                                 side=OrderSide.BUY if o["side"] == "buy" else OrderSide.SELL,
                                 time_in_force=TimeInForce.DAY)
        resp = trading_client.submit_order(req)
        out.append({**o, "order_id": str(resp.id), "status": str(resp.status)})
    return out


# ---------------------------------------------------------------------------
# Registro y aviso
# ---------------------------------------------------------------------------

def notify(text: str):
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        return
    try:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", json={"chat_id": chat, "text": text}, timeout=10)
    except Exception as e:
        print("telegram error:", e)


def log(name: str, record: dict):
    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR / f"{datetime.now(timezone.utc):%Y-%m-%d}_{name}.json").write_text(json.dumps(record, indent=2, default=str))


# ---------------------------------------------------------------------------

def run(name: str, force: bool, dry_run: bool):
    s = STRATEGIES[name]
    trading, data = alpaca_clients(name)

    if not force and not is_first_trading_day_of_month(trading):
        print(f"[{name}] Hoy no es el primer día hábil del mes. Nada que hacer.")
        return

    prices = fetch_prices(data, symbols_needed(s), days=max(s.lookback, s.vol_window) + 40)
    scores = momentum_scores(prices, s.lookback)
    targets = target_weights(prices, s)
    orders = plan_orders(trading, targets)
    equity = float(trading.get_account().equity)

    msg = "\n".join([
        f"{name} — rebalanceo {datetime.now(timezone.utc):%Y-%m-%d}",
        f"Equity: ${equity:,.0f}",
        f"Scores {s.lookback}d: " + ", ".join(f"{k} {v:+.1%}" for k, v in scores.sort_values(ascending=False).items()),
        "Objetivo: " + (", ".join(f"{k} {v:.0%}" for k, v in targets[targets > 0].items()) or "100% efectivo")
        + (f"  (exposición {targets.sum():.0%})" if s.vol_target else ""),
        "Órdenes: " + (", ".join(f"{o['side']} {o['symbol']} ${o['notional']:,.0f}" for o in orders) or "ninguna"),
    ])
    print(msg)

    results = [] if dry_run else submit(trading, orders)
    if dry_run:
        print("(dry-run: no se envió nada)")
    log(name, {"date": str(datetime.now(timezone.utc)), "strategy": name, "equity": equity,
               "last_price_date": str(prices.index[-1].date()), "scores": scores.round(4).to_dict(),
               "targets": targets.round(4).to_dict(), "orders": results or orders, "dry_run": dry_run})
    if not dry_run:
        notify(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=list(STRATEGIES), default="MOM6")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    try:
        run(a.strategy, a.force, a.dry_run)
    except Exception as e:
        notify(f"{a.strategy} — ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
