"""Live-market tabular task — verifies GBDT-vs-MLP beyond synthetic data.

Spot values verified 2026-10-16 via openresearch tools before coding:
  BTC/USD ~80,833.80 (7d series 82.5k->80.8k), USD->EUR 0.88865, GBP 0.74989, JPY 158.02.
Protocol: fetch BTC daily + FX daily (CoinGecko + Frankfurter, no keys),
build lag/return/vol features -> predict next-day log-return (regression).
Falls back to seeded random-walk if offline so Docker never breaks.
"""
import numpy as np

BTC_SPOT_VERIFY = {"date": "2026-10-16", "close": 80833.80}
FX_SPOT_VERIFY = {"date": "2026-10-16", "EUR": 0.88865, "GBP": 0.74989, "JPY": 158.02}


def fetch_btc(days=365):
    try:
        import urllib.request, json
        url = ("https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
               f"?vs_currency=usd&days={days}&interval=daily")
        with urllib.request.urlopen(url, timeout=20) as r:
            j = json.load(r)
        px = np.array([p[1] for p in j["prices"]], dtype=float)
        if len(px) > 20:
            return px
    except Exception as e:
        print(f"[live] CoinGecko failed ({e}), using fallback walk")
    rng = np.random.default_rng(7)
    ret = rng.normal(0.001, 0.03, days)
    return 80000 * np.exp(np.cumsum(ret))


def fetch_fx(sym="EUR", days=365):
    try:
        import urllib.request, json
        from datetime import date, timedelta
        end = date(2026, 10, 16)
        start = end - timedelta(days=days * 2)
        url = (f"https://api.frankfurter.dev/v1/{start}:{end}"
               f"?base=USD&symbols={sym}")
        with urllib.request.urlopen(url, timeout=20) as r:
            j = json.load(r)
        rates = [v[sym] for k, v in sorted(j["rates"].items()) if sym in v]
        px = np.array(rates, dtype=float)
        if len(px) > 20:
            return px[-days:]
    except Exception as e:
        print(f"[live] Frankfurter failed ({e}), using fallback walk")
    rng = np.random.default_rng(9)
    return 0.89 * np.exp(np.cumsum(rng.normal(0, 0.003, days)))


def to_tabular(price, lags=(1, 2, 3, 5, 7), vol_w=7):
    lr = np.diff(np.log(price))
    X, y = [], []
    for t in range(max(lags) + vol_w, len(lr) - 1):
        f = [lr[t - l] for l in lags]
        w = lr[t - vol_w:t]
        f += [w.mean(), w.std(), w.min(), w.max()]
        X.append(f)
        y.append(lr[t + 1])
    names = [f"lag{l}" for l in lags] + ["m", "sd", "mn", "mx"]
    return np.array(X), np.array(y), names
