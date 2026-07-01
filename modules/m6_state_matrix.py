"""
MODULO 6 - STATE MATRIX POR MOVIMIENTO
Foto de todos los indicadores en inicio y fin de cada movimiento.
Outputs:
    data/processed/{symbol}_state_at_start.csv
    data/processed/{symbol}_state_at_end.csv
"""
import os
import pandas as pd
import numpy as np
from config import DATA_DIR

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SKIP_COLS = {"open", "high", "low", "close", "volume", "swing_type", "swing_price"}


def _state_at(ts, tfs_ind):
    state = {}
    for tf_name, df in tfs_ind.items():
        subset = df.sort_index()
        subset = subset[subset.index <= ts]
        if subset.empty:
            continue
        row = subset.iloc[-1]
        for col in df.columns:
            if col not in SKIP_COLS:
                v = row.get(col, np.nan)
                state[f"{tf_name}_{col}"] = v
    return state


def run_state_matrix(symbol, movements, tfs_ind):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if movements.empty:
        print("  Sin movimientos.")
        return pd.DataFrame(), pd.DataFrame()

    rows_s, rows_e = [], []
    total = len(movements)
    for i, (_, mov) in enumerate(movements.iterrows()):
        if i % 200 == 0:
            print(f"  State matrix {i+1}/{total}...")
        ts_ini = pd.to_datetime(mov["timestamp_inicio"], utc=True)
        ts_fin = pd.to_datetime(mov["timestamp_fin"], utc=True) if pd.notna(mov.get("timestamp_fin")) else None
        meta = {"mov_id": mov["mov_id"], "tf": mov["tf"], "direccion": mov["direccion"],
                "magnitud_pct": mov.get("magnitud_pct", np.nan),
                "duracion_min": mov.get("duracion_min", np.nan),
                "tf_parent_id": mov.get("tf_parent_id", None),
                "timestamp_inicio": ts_ini}
        rows_s.append({**meta, **_state_at(ts_ini, tfs_ind)})
        if ts_fin is not None:
            rows_e.append({**meta, **_state_at(ts_fin, tfs_ind)})

    df_s = pd.DataFrame(rows_s)
    df_e = pd.DataFrame(rows_e)
    df_s.to_csv(os.path.join(PROCESSED_DIR, f"{symbol}_state_at_start.csv"), index=False)
    df_e.to_csv(os.path.join(PROCESSED_DIR, f"{symbol}_state_at_end.csv"),   index=False)
    print(f"  state_at_start: {len(df_s)} filas")
    print(f"  state_at_end  : {len(df_e)} filas")
    return df_s, df_e
