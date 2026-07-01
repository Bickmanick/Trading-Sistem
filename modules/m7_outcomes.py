"""
MODULO 7 - OUTCOMES
Para cada movimiento calcula:
  - MFE / MAE dentro del propio movimiento (velas 1M entre inicio y fin)
  - ratio_mfe_mae = calidad del movimiento (>2 = limpio, <0.5 = ruidoso)
  - rebote_fib = nivel Fibonacci del swing PREVIO donde termino el movimiento
  - continuacion = CONTINUACION si el siguiente mov del mismo TF va en la misma
                   direccion, REVERSION si va en direccion contraria.
                   Criterio secundario: ratio_mfe_mae >= 2.0 refuerza CONTINUACION

Output: data/processed/{symbol}_outcomes.csv
"""
import os
import pandas as pd
import numpy as np
from config import DATA_DIR

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FIB_LEVELS    = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]


def _nearest_fib(ratio: float) -> float:
    return min(FIB_LEVELS, key=lambda f: abs(f - ratio))


def _mfe_mae(mov: pd.Series, df_1m: pd.DataFrame) -> tuple:
    """MFE y MAE calculados sobre las velas 1M dentro del propio movimiento."""
    ts_ini = pd.to_datetime(mov["timestamp_inicio"], utc=True)
    ts_fin = pd.to_datetime(mov["timestamp_fin"],    utc=True)
    w = df_1m[(df_1m.index >= ts_ini) & (df_1m.index <= ts_fin)]
    if w.empty:
        return np.nan, np.nan
    ep   = mov["precio_inicio"]
    dir_ = mov["direccion"]
    if dir_ == "LONG":
        mfe = (w["high"].max() - ep) / ep * 100
        mae = (ep - w["low"].min())  / ep * 100
    else:
        mfe = (ep - w["low"].min())  / ep * 100
        mae = (w["high"].max() - ep) / ep * 100
    return round(float(max(mfe, 0)), 4), round(float(max(mae, 0)), 4)


def _rebote_fib_sobre_swing_previo(mov: pd.Series, movs_tf: pd.DataFrame) -> float:
    """
    Mide en que nivel Fibonacci del movimiento PREVIO del mismo TF
    termino el movimiento actual.
    - Si el movimiento previo es LONG:  rango = [precio_inicio_prev, precio_extremo_prev]
      La reversion SHORT deberia caer hacia ese rango.
    - Si el movimiento previo es SHORT: rango = [precio_extremo_prev, precio_inicio_prev]
      La reversion LONG deberia subir hacia ese rango.
    rebote_fib = cuanto retrocedio dentro de ese rango (0.0 = no retrocedio, 1.0 = llego al origen)
    """
    ts_ini = pd.to_datetime(mov["timestamp_inicio"], utc=True)
    previos = movs_tf[movs_tf["timestamp_fin"] < ts_ini].copy()
    if previos.empty:
        return np.nan
    prev = previos.sort_values("timestamp_fin").iloc[-1]
    pi_prev = prev["precio_inicio"]
    pe_prev = prev["precio_extremo"]
    rango   = abs(pe_prev - pi_prev)
    if rango == 0 or pd.isna(pe_prev) or pd.isna(pi_prev):
        return np.nan
    precio_fin_actual = mov.get("precio_fin", np.nan)
    if pd.isna(precio_fin_actual):
        return np.nan
    if prev["direccion"] == "LONG":
        # retroceso desde el extremo del mov previo hacia su inicio
        ret = pe_prev - precio_fin_actual
    else:
        # retroceso desde el extremo del mov previo hacia su inicio
        ret = precio_fin_actual - pe_prev
    ratio = ret / rango
    ratio = max(0.0, min(1.0, ratio))
    return _nearest_fib(ratio)


def _continuacion(mov: pd.Series, movs_tf: pd.DataFrame) -> str:
    """
    CONTINUACION si el siguiente movimiento del mismo TF va en la misma direccion.
    REVERSION   si va en direccion contraria.
    ULTIMO      si no hay movimiento siguiente (ultimo del historial).
    """
    ts_fin = pd.to_datetime(mov["timestamp_fin"], utc=True)
    sig    = movs_tf[movs_tf["timestamp_inicio"] > ts_fin].copy()
    if sig.empty:
        return "ULTIMO"
    siguiente = sig.sort_values("timestamp_inicio").iloc[0]
    if siguiente["direccion"] == mov["direccion"]:
        return "CONTINUACION"
    return "REVERSION"


def run_outcomes(symbol: str, movements: pd.DataFrame, df_1m: pd.DataFrame) -> pd.DataFrame:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if movements.empty:
        print("  Sin movimientos.")
        return pd.DataFrame()

    df_1m = df_1m.sort_index()
    movements = movements.copy()
    movements["timestamp_inicio"] = pd.to_datetime(movements["timestamp_inicio"], utc=True)
    movements["timestamp_fin"]    = pd.to_datetime(movements["timestamp_fin"],    utc=True)

    # Filtrar solo movimientos con fin conocido
    validos = movements.dropna(subset=["timestamp_fin"])

    rows  = []
    total = len(validos)
    for i, (_, mov) in enumerate(validos.iterrows()):
        if i % 500 == 0:
            print(f"  Outcomes {i+1}/{total}...")

        # Movimientos del mismo TF para comparacion
        movs_tf = validos[validos["tf"] == mov["tf"]].copy()

        mfe, mae = _mfe_mae(mov, df_1m)
        ratio    = round(mfe / mae, 3) if (mae and mae > 0) else np.nan
        fib      = _rebote_fib_sobre_swing_previo(mov, movs_tf)
        cont     = _continuacion(mov, movs_tf)

        rows.append({
            "mov_id":        mov["mov_id"],
            "tf":            mov["tf"],
            "direccion":     mov["direccion"],
            "magnitud_pct":  round(float(mov.get("magnitud_pct", 0)), 4),
            "duracion_min":  mov.get("duracion_min", np.nan),
            "mfe_pct":       mfe,
            "mae_pct":       mae,
            "ratio_mfe_mae": ratio,
            "rebote_fib":    fib,
            "continuacion":  cont,
        })

    df_out = pd.DataFrame(rows)
    path   = os.path.join(PROCESSED_DIR, f"{symbol}_outcomes.csv")
    df_out.to_csv(path, index=False)
    print(f"  Outcomes: {len(df_out)} filas -> {path}")
    return df_out
