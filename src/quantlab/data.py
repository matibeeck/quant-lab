"""
Descarga y cachea precios de cierre ajustados.

Usamos 'Adj Close' (cierre ajustado) y no 'Close' porque el ajustado
incorpora dividendos y splits. Sin eso, el día que una empresa hace un
split 2:1 verías una "caída" del 50% que nunca ocurrió.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_prices(
    tickers: list[str],
    start: str = "2019-01-01",
    end: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Devuelve un DataFrame: índice = fechas, columnas = tickers, valores = precio ajustado.
    Guarda un parquet en /data para no descargar cada vez.
    """
    key = "_".join(sorted(tickers)) + f"_{start}_{end or 'today'}.parquet"
    path = DATA_DIR / key

    if cache and path.exists():
        return pd.read_parquet(path)

    import yfinance as yf  # import aquí para que el módulo cargue sin red

    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    prices = raw["Adj Close"]
    if isinstance(prices, pd.Series):  # un solo ticker
        prices = prices.to_frame(tickers[0])
    prices = prices.dropna(how="all")

    if cache:
        DATA_DIR.mkdir(exist_ok=True)
        prices.to_parquet(path)
    return prices
