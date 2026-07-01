"""
MODULO 7 - OUTCOMES

Por cada movimiento calcula:
  - MFE / MAE dentro del propio movimiento (velas 1M entre inicio y fin)
  - ratio_mfe_mae  (>2 limpio, <0.5 ruidoso)
  - rebote_fib     nivel Fibonacci del swing PREVIO donde termino el movimiento
  - fib_agotamiento nivel Fibonacci del PROPIO movimiento donde retrocedio antes de cerrar
  - continuacion   CONTINUACION / REVERSION / ULTIMO segun el siguiente mov del mismo TF
  - alineamiento_3tf True si el movimiento y sus dos TF padres van en la misma direccion

Output: data/processed/{symbol}_outcomes.csv
"""
import os
import pandas as pd
import numpy as np
from config import DATA_DIR

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FIB_LEVELS    = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
TF_ORDER      = ["1M", "5M", "15M", "30M", "1H", "4H", "1D", "1W"]


def _nearest_fib(ratio: float) -> float:
    return min(FIB_LEVELS, key=lambda f: abs(f - ratio))


def _mfe_mae(mov: pd.Series, df_1m: pd.DataFrame) -> tuple:
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


def _rebote_fib_swing_previo(mov: pd.Series, movs_tf: pd.DataFrame) -> float:
    """Nivel Fibonacci del swing PREVIO donde termino el movimiento actual."""
    ts_ini  = pd.to_datetime(mov["timestamp_inicio"], utc=True)
    previos = movs_tf[movs_tf["timestamp_fin"] < ts_ini]
    if previos.empty:
        return np.nan
    prev   = previos.sort_values("timestamp_fin").iloc[-1]
    pi_p   = prev["precio_inicio"]
    pe_p   = prev["precio_extremo"]
    rango  = abs(pe_p - pi_p)
    if rango == 0 or pd.isna(pe_p) or pd.isna(pi_p):
        return np.nan
    pf = mov.get("precio_fin", np.nan)
    if pd.isna(pf):
        return np.nan
    ret   = (pe_p - pf) if prev["direccion"] == "LONG" else (pf - pe_p)
    ratio = max(0.0, min(1.0, ret / rango))
    return _nearest_fib(ratio)


def _fib_agotamiento(mov: pd.Series, df_1m: pd.DataFrame) -> float:
    """
    Fibonacci del PROPIO movimiento: hasta que nivel retrocedio el precio
    desde el extremo antes de cerrar.
    Indica donde se agoto el movimiento internamente.
    """
    ts_ini = pd.to_datetime(mov["timestamp_inicio"], utc=True)
    ts_fin = pd.to_datetime(mov["timestamp_fin"],    utc=True)
    w = df_1m[(df_1m.index >= ts_ini) & (df_1m.index <= ts_fin)]
    if w.empty:
        return np.nan
    pi  = mov["precio_inicio"]
    pe  = mov["precio_extremo"]
    pf  = mov.get("precio_fin", np.nan)
    rango = abs(pe - pi)
    if rango == 0 or pd.isna(pf):
        return np.nan
    if mov["direccion"] == "LONG":
        ret = pe - pf   # cuanto retrocedio desde el maximo hasta el cierre
    else:
        ret = pf - pe   # cuanto retrocedio desde el minimo hasta el cierre
    ratio = max(0.0, min(1.0, ret / rango))
    return _nearest_fib(ratio)


def _continuacion(mov: pd.Series, movs_tf: pd.DataFrame) -> str:
    ts_fin = pd.to_datetime(mov["timestamp_fin"], utc=True)
    sig    = movs_tf[movs_tf["timestamp_inicio"] > ts_fin]
    if sig.empty:
        return "ULTIMO"
    siguiente = sig.sort_values("timestamp_inicio").iloc[0]
    return "CONTINUACION" if siguiente["direccion"] == mov["direccion"] else "REVERSION"


def _alineamiento_3tf(mov: pd.Series, todos: pd.DataFrame) -> bool:
    """
    True si el movimiento + su padre TF + el abuelo TF van en la misma direccion.
    Requiere que tf_parent_id este poblado por m5.
    """
    dir_  = mov["direccion"]
    pid   = mov.get("tf_parent_id")
    if pd.isna(pid) or not pid:
        return False
    parent = todos[todos["mov_id"] == pid]
    if parent.empty or parent.iloc[0]["direccion"] != dir_:
        return False
    gpid = parent.iloc[0].get("tf_parent_id")
    if pd.isna(gpid) or not gpid:
        return True   # alineamiento de 2 TFs al menos
    grand = todos[todos["mov_id"] == gpid]
    if grand.empty:
        return True
    return grand.iloc[0]["direccion"] == dir_


def run_outcomes(symbol: str, movements: pd.DataFrame,
                 df_1m: pd.DataFrame) -> pd.DataFrame:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if movements.empty:
        print("  Sin movimientos.")
        return pd.DataFrame()

    df_1m = df_1m.sort_index()
    movements = movements.copy()
    movements["timestamp_inicio"] = pd.to_datetime(movements["timestamp_inicio"], utc=True)
    movements["timestamp_fin"]    = pd.to_datetime(movements["timestamp_fin"],    utc=True)

    validos = movements.dropna(subset=["timestamp_fin"]).copy()

    rows  = []
    total = len(validos)
    for i, (_, mov) in enumerate(validos.iterrows()):
        if i % 500 == 0:
            print(f"  Outcomes {i+1}/{total}...")

        movs_tf = validos[validos["tf"] == mov["tf"]]

        mfe, mae  = _mfe_mae(mov, df_1m)
        ratio     = round(mfe / mae, 3) if (mae and mae > 0) else np.nan
        fib_prev  = _rebote_fib_swing_previo(mov, movs_tf)
        fib_agot  = _fib_agotamiento(mov, df_1m)
        cont      = _continuacion(mov, movs_tf)
        alin      = _alineamiento_3tf(mov, validos)

        rows.append({
            "mov_id":           mov["mov_id"],
            "tf":               mov["tf"],
            "direccion":        mov["direccion"],
            "magnitud_pct":     round(float(mov.get("magnitud_pct", 0)), 4),
            "duracion_min":     mov.get("duracion_min", np.nan),
            "cierre_tipo":      mov.get("cierre_tipo", np.nan),
            "mfe_pct":          mfe,
            "mae_pct":          mae,
            "ratio_mfe_mae":    ratio,
            "rebote_fib":       fib_prev,
            "fib_agotamiento":  fib_agot,
            "continuacion":     cont,
            "alineamiento_3tf": alin,
        })

    df_out = pd.DataFrame(rows)
    path   = os.path.join(PROCESSED_DIR, f"{symbol}_outcomes.csv")
    df_out.to_csv(path, index=False)
    print(f"  Outcomes: {len(df_out)} filas -> {path}")
    return df_out
