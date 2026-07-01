"""
MODULO 6 - STATE MATRIX POR MOVIMIENTO (version rapida)
En lugar de reconstruir el estado barra a barra, usa el state matrix
completo ya calculado por state_engine.build_state_matrix() y hace
un simple merge_asof para encontrar el estado en cada timestamp.

Tiempo: <10 segundos (vs 30 minutos del loop anterior).

Outputs:
    data/processed/{symbol}_state_at_start.csv
    data/processed/{symbol}_state_at_end.csv
"""
import os
import pandas as pd
import numpy as np
from config import DATA_DIR

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def run_state_matrix(symbol: str, movements: pd.DataFrame,
                     state_matrix: pd.DataFrame) -> tuple:
    """
    Parameters
    ----------
    symbol       : ticker
    movements    : DataFrame de m5 con columnas timestamp_inicio / timestamp_fin
    state_matrix : DataFrame devuelto por state_engine.build_state_matrix()
                   indexado por timestamp 1M con todas las columnas TF_col
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if movements.empty:
        print("  Sin movimientos.")
        return pd.DataFrame(), pd.DataFrame()

    # Asegurar timezone UTC en state_matrix
    sm = state_matrix.copy()
    if sm.index.tz is None:
        sm.index = sm.index.tz_localize("UTC")
    sm = sm.sort_index()

    # Columnas de estado (excluir OHLC base)
    ohlc = {"open", "high", "low", "close"}
    state_cols = [c for c in sm.columns if c not in ohlc]

    # Timestamps de inicio y fin como Series UTC
    ts_ini = pd.to_datetime(movements["timestamp_inicio"], utc=True)
    ts_fin = pd.to_datetime(movements["timestamp_fin"],    utc=True)

    meta_cols = ["mov_id", "tf", "direccion", "magnitud_pct",
                 "duracion_min", "tf_parent_id"]
    meta = movements[meta_cols].copy()
    meta["timestamp_inicio"] = ts_ini.values

    def _lookup(timestamps: pd.Series) -> pd.DataFrame:
        """
        Para cada timestamp, busca la fila mas reciente en state_matrix
        usando merge_asof (equivalente a ffill por timestamp).
        """
        ts_df = pd.DataFrame({"ts": timestamps.values})
        ts_df["ts"] = pd.to_datetime(ts_df["ts"], utc=True)
        ts_df = ts_df.sort_values("ts").reset_index(drop=True)

        sm_reset = sm[state_cols].copy()
        sm_reset.index.name = "ts"
        sm_reset = sm_reset.reset_index()

        merged = pd.merge_asof(
            ts_df, sm_reset,
            on="ts", direction="backward"
        )
        return merged

    print(f"  Lookup inicio ({len(movements)} movimientos)...")
    starts = _lookup(ts_ini)
    starts.index = movements.index
    df_start = pd.concat([meta, starts[state_cols]], axis=1)

    print(f"  Lookup fin ({ts_fin.notna().sum()} movimientos con fin)...")
    mask_fin  = ts_fin.notna()
    if mask_fin.any():
        ends = _lookup(ts_fin[mask_fin])
        ends.index = movements[mask_fin].index
        meta_fin  = meta[mask_fin].copy()
        df_end = pd.concat([meta_fin, ends[state_cols]], axis=1)
    else:
        df_end = pd.DataFrame()

    path_s = os.path.join(PROCESSED_DIR, f"{symbol}_state_at_start.csv")
    path_e = os.path.join(PROCESSED_DIR, f"{symbol}_state_at_end.csv")
    df_start.to_csv(path_s, index=False)
    df_end.to_csv(path_e,   index=False)
    print(f"  state_at_start: {len(df_start)} filas -> {path_s}")
    print(f"  state_at_end  : {len(df_end)} filas -> {path_e}")
    return df_start, df_end
