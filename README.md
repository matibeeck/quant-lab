# quant-lab

Laboratorio personal de finanzas cuantitativas, construido desde cero.
Un proyecto por concepto; cada uno reutiliza la librería `quantlab` de `src/`.

## Setup (una vez)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # instala quantlab en modo editable
pytest                           # todos los tests deben pasar
jupyter lab                      # abre notebooks/01_retornos_y_riesgo.ipynb
```

## Estructura

```
src/quantlab/     librería reutilizable (metrics.py, data.py) — crece con cada proyecto
notebooks/        un notebook guiado por proyecto
tests/            tests de la librería con casos conocidos a mano
data/             cache de precios (parquet, ignorado por git)
```

## Roadmap

| # | Proyecto | Conceptos | Estado |
|---|----------|-----------|--------|
| 1 | Retornos y riesgo | retornos simples/log, colas gordas, volatilidad, drawdown, Sharpe, correlación | hecho |
| 2 | Monte Carlo de precios | GBM, bootstrap, VaR/CVaR, correlación vía Cholesky | hecho |
| 3 | Series de tiempo | autocorrelación, estacionariedad, GARCH a mano, régimen de volatilidad | en curso |
| 4 | Backtester propio | señales, costos, look-ahead bias, walk-forward | |
| 5 | Pricing de opciones | Black-Scholes, binomial, Monte Carlo, griegas | |
| 6 | Optimización de portafolios | Markowitz, frontera eficiente, risk parity, shrinkage | |
| 7 | ML financiero | purged CV, meta-labeling, feature importance | |

## Capstone

Los proyectos 3-7 alimentan una app propia (Quant Lab): riesgo de cartera, alertas de régimen de volatilidad, estrategia en paper trading con avisos por Telegram, pricing de opciones y optimizador. Cada proyecto entrega un notebook interactivo (ipywidgets) y su módulo en `quantlab/`.

## Paper trading (`paper/`)

Tres variantes de momentum mensual sobre 8 ETFs, cada una en su propia cuenta paper de Alpaca,
ejecutadas por GitHub Actions el primer día hábil de cada mes:

| Nombre | Regla | Referencia |
|---|---|---|
| MOM6 | top 3 por retorno 6m, si > 0; resto efectivo | Faber (2007) |
| VOLT | MOM6 con exposición escalada a 10% de vol anual | Moreira & Muir (2017) |
| DUAL | top 3 por retorno 12m, si > retorno de T-bills (BIL) | Antonacci (2014) |

`paper/backtest_check.py` reproduce las tres con datos de Yahoo desde 2007. Backtest y operación
usan la misma función `target_weights`.

## Reglas

- Primero construir a mano, luego comparar con librerías.
- Toda función nueva en `quantlab` lleva un test con un caso resuelto a mano.
- Commit al terminar cada sesión.
