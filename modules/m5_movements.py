"""
MODULO 5 - DETECTOR DE MOVIMIENTOS
Usa los CHoCH del m4_structure para definir tramos LONG/SHORT reales
y los vincula entre TFs padre-hijo.
Outputs:
    data/processed/{symbol}_movements_{tf}.csv
    data/processed/{symbol}_movements_linked.csv
"""
import os
import pandas as pd
import numpy as np
from config import DATA_DIR

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
TF_MINUTES = {"1M": 1, "5M": 5, "15M": 15, "30M": 30, "1H": 60, "4H": 240, "1D": 1440, "1W": 10080}
TF_ORDER   = ["1M", "5M", "15M", "30M", "1H", "4H", "1D", "1W"]


def _extract_movements(df_struct, tf_name):
    rows, mov_id, active = [], 0, None
    tf_mins = TF_MINUTES.get(tf_name, 1)

    for idx, row in df_struct.iterrows():
        ce = row.get("choch_event", "NONE")
        if active and ce == "NONE":
            if active["direccion"] == "LONG":
                active["precio_extremo"] = max(active["precio_extremo"],
                                               row.get("high", active["precio_extremo"]))
            else:
                active["precio_extremo"] = min(active["precio_extremo"],
                                               row.get("low", active["precio_extremo"]))
            active["duracion_barras"] += 1
            continue

        if ce == "CHOCH_ALCISTA":
            if active and active["direccion"] == "SHORT":
                active.update({"timestamp_fin": idx, "precio_fin": row.get("close", np.nan)})
                pi, pe = active["precio_inicio"], active["precio_extremo"]
                active["magnitud_pct"] = round((pi - pe) / pi * 100, 4) if pi else 0
                active["duracion_min"] = active["duracion_barras"] * tf_mins
                rows.append(active)
            mov_id += 1
            active = {"mov_id": f"{tf_name}_{mov_id}", "tf": tf_name, "direccion": "LONG",
                      "timestamp_inicio": idx, "timestamp_fin": None,
                      "precio_inicio": row.get("close", np.nan), "precio_fin": np.nan,
                      "precio_extremo": row.get("high", row.get("close", 0)),
                      "magnitud_pct": 0.0, "duracion_barras": 1, "duracion_min": 0, "tf_parent_id": None}

        elif ce == "CHOCH_BAJISTA":
            if active and active["direccion"] == "LONG":
                active.update({"timestamp_fin": idx, "precio_fin": row.get("close", np.nan)})
                pi, pe = active["precio_inicio"], active["precio_extremo"]
                active["magnitud_pct"] = round((pe - pi) / pi * 100, 4) if pi else 0
                active["duracion_min"] = active["duracion_barras"] * tf_mins
                rows.append(active)
            mov_id += 1
            active = {"mov_id": f"{tf_name}_{mov_id}", "tf": tf_name, "direccion": "SHORT",
                      "timestamp_inicio": idx, "timestamp_fin": None,
                      "precio_inicio": row.get("close", np.nan), "precio_fin": np.nan,
                      "precio_extremo": row.get("low", row.get("close", 9999999)),
                      "magnitud_pct": 0.0, "duracion_barras": 1, "duracion_min": 0, "tf_parent_id": None}

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _link_movements(all_movs):
    frames = [df.copy() for df in all_movs.values() if not df.empty]
    if not frames:
        return pd.DataFrame()
    linked = pd.concat(frames, ignore_index=True)
    linked["timestamp_inicio"] = pd.to_datetime(linked["timestamp_inicio"], utc=True)
    linked["timestamp_fin"]    = pd.to_datetime(linked["timestamp_fin"],    utc=True)
    for i, row in linked.iterrows():
        tf_idx = TF_ORDER.index(row["tf"]) if row["tf"] in TF_ORDER else -1
        if tf_idx < 0 or tf_idx >= len(TF_ORDER) - 1:
            continue
        parents = linked[
            (linked["tf"] == TF_ORDER[tf_idx + 1]) &
            (linked["timestamp_inicio"] <= row["timestamp_inicio"]) &
            (linked["timestamp_fin"].isna() | (linked["timestamp_fin"] >= row["timestamp_inicio"]))
        ]
        if not parents.empty:
            linked.at[i, "tf_parent_id"] = parents.iloc[-1]["mov_id"]
    return linked


def run_movements(symbol, structure_data):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    all_movs = {}
    for tf_name, df_struct in structure_data.items():
        print(f"  [{tf_name}] Extrayendo movimientos...")
        df_mov = _extract_movements(df_struct, tf_name)
        if not df_mov.empty:
            df_mov.to_csv(os.path.join(PROCESSED_DIR, f"{symbol}_movements_{tf_name}.csv"), index=False)
            print(f"  [{tf_name}] {len(df_mov)} movimientos detectados")
        all_movs[tf_name] = df_mov
    linked = _link_movements(all_movs)
    if not linked.empty:
        linked.to_csv(os.path.join(PROCESSED_DIR, f"{symbol}_movements_linked.csv"), index=False)
        print(f"  Total vinculados: {len(linked)} movimientos")
    return linked
