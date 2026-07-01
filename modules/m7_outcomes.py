"""
MODULO 7 - OUTCOMES
MFE, MAE, rebote Fibonacci, continuacion/reversion por movimiento.
Output: data/processed/{symbol}_outcomes.csv
"""
import os
import pandas as pd
import numpy as np
from config import DATA_DIR

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FIB_LEVELS    = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]


def _nearest_fib(ratio):
    return min(FIB_LEVELS, key=lambda f: abs(f - ratio))


def _outcome(mov, df_1m):
    ts_ini = pd.to_datetime(mov["timestamp_inicio"], utc=True)
    ts_fin = pd.to_datetime(mov["timestamp_fin"],    utc=True) if pd.notna(mov.get("timestamp_fin")) else None
    if ts_fin is None:
        return {}
    w = df_1m[(df_1m.index >= ts_ini) & (df_1m.index <= ts_fin)]
    if w.empty:
        return {}

    ep, dir_ = mov["precio_inicio"], mov["direccion"]
    mfe = (w["high"].max() - ep) / ep * 100 if dir_ == "LONG" else (ep - w["low"].min()) / ep * 100
    mae = (ep - w["low"].min()) / ep * 100  if dir_ == "LONG" else (w["high"].max() - ep) / ep * 100

    post = df_1m[df_1m.index > ts_fin].head(60)
    rebote_fib, continuacion = np.nan, "DESCONOCIDO"
    pe    = mov.get("precio_extremo", np.nan)
    rango = abs(pe - ep) if pd.notna(pe) else 0

    if not post.empty and rango > 0:
        if dir_ == "LONG":
            ret          = max(0, pe - post["low"].min())
            rebote_fib   = _nearest_fib(min(1, ret / rango))
            continuacion = "CONTINUACION" if post["close"].iloc[-1] > pe else "REVERSION"
        else:
            ret          = max(0, post["high"].max() - pe)
            rebote_fib   = _nearest_fib(min(1, ret / rango))
            continuacion = "CONTINUACION" if post["close"].iloc[-1] < pe else "REVERSION"

    return {"mov_id": mov["mov_id"], "tf": mov["tf"], "direccion": dir_,
            "magnitud_pct": round(float(mov.get("magnitud_pct", 0)), 4),
            "duracion_min": mov.get("duracion_min", np.nan),
            "mfe_pct":  round(float(mfe), 4), "mae_pct": round(float(mae), 4),
            "ratio_mfe_mae": round(float(mfe/mae), 3) if mae > 0 else np.nan,
            "rebote_fib": rebote_fib, "continuacion": continuacion}


def run_outcomes(symbol, movements, df_1m):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if movements.empty:
        print("  Sin movimientos.")
        return pd.DataFrame()
    df_1m = df_1m.sort_index()
    rows  = []
    total = len(movements)
    for i, (_, mov) in enumerate(movements.iterrows()):
        if i % 500 == 0:
            print(f"  Outcomes {i+1}/{total}...")
        out = _outcome(mov, df_1m)
        if out:
            rows.append(out)
    df_out = pd.DataFrame(rows)
    path   = os.path.join(PROCESSED_DIR, f"{symbol}_outcomes.csv")
    df_out.to_csv(path, index=False)
    print(f"  Outcomes: {len(df_out)} filas -> {path}")
    return df_out
