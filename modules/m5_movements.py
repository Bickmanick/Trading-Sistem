"""
MODULO 5 - DETECTOR DE MOVIMIENTOS

Criterios de INICIO de movimiento:
  1. CHoCH confirmado (cambio de estructura)
  2. BOS en nueva direccion (impulso sin CHoCH previo)

Criterios de FIN de movimiento (el primero que ocurra):
  1. CHoCH contrario confirmado
  2. BOS contrario
  3. Agotamiento: retroceso >= umbral_tf desde el precio_extremo

Filtro minimo: duracion >= 3 barras del TF

Outputs:
    data/processed/{symbol}_movements_{tf}.csv
    data/processed/{symbol}_movements_linked.csv
"""
import os
import pandas as pd
import numpy as np
from config import DATA_DIR

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

TF_MINUTES = {"1M": 1, "5M": 5, "15M": 15, "30M": 30,
               "1H": 60, "4H": 240, "1D": 1440, "1W": 10080}
TF_ORDER   = ["1M", "5M", "15M", "30M", "1H", "4H", "1D", "1W"]

# Retroceso minimo desde el extremo para considerar agotamiento
EXHAUSTION_PCT = {
    "1M":  0.30, "5M":  0.50, "15M": 0.80, "30M": 1.00,
    "1H":  1.50, "4H":  2.50, "1D":  4.00, "1W":  6.00,
}
# Duracion minima en barras para que un movimiento sea valido
MIN_BARS = 3


def _close_active(active: dict, idx, row, tf_mins: int) -> dict:
    """Cierra el movimiento activo con los datos de la barra actual."""
    active["timestamp_fin"] = idx
    active["precio_fin"]    = row.get("close", np.nan)
    pi = active["precio_inicio"]
    pe = active["precio_extremo"]
    if active["direccion"] == "LONG":
        active["magnitud_pct"] = round((pe - pi) / pi * 100, 4) if pi else 0
    else:
        active["magnitud_pct"] = round((pi - pe) / pi * 100, 4) if pi else 0
    active["duracion_min"] = active["duracion_barras"] * tf_mins
    return active


def _extract_movements(df_struct: pd.DataFrame, tf_name: str) -> pd.DataFrame:
    rows, mov_id, active = [], 0, None
    tf_mins  = TF_MINUTES.get(tf_name, 1)
    exh_pct  = EXHAUSTION_PCT.get(tf_name, 1.5)

    for idx, row in df_struct.iterrows():
        ce  = row.get("choch_event", "NONE")
        bos = row.get("bos_event",   "NONE")
        close = row.get("close", np.nan)

        # --- Actualizar extremo del movimiento activo ---
        if active:
            if active["direccion"] == "LONG":
                if row.get("high", active["precio_extremo"]) > active["precio_extremo"]:
                    active["precio_extremo"] = row["high"]
            else:
                if row.get("low", active["precio_extremo"]) < active["precio_extremo"]:
                    active["precio_extremo"] = row["low"]
            active["duracion_barras"] += 1

            # Criterio 3: agotamiento por retroceso desde extremo
            pe = active["precio_extremo"]
            if pe and pe > 0:
                if active["direccion"] == "LONG":
                    retroceso = (pe - close) / pe * 100
                else:
                    retroceso = (close - pe) / pe * 100
                if retroceso >= exh_pct and active["duracion_barras"] >= MIN_BARS:
                    active = _close_active(active, idx, row, tf_mins)
                    active["cierre_tipo"] = "AGOTAMIENTO"
                    rows.append(active)
                    active = None
                    # No abrir nuevo movimiento — esperamos CHoCH o BOS
                    continue

            # Criterio 2: BOS contrario cierra el movimiento activo
            if active:
                if active["direccion"] == "LONG" and bos == "BOS_BEAR":
                    if active["duracion_barras"] >= MIN_BARS:
                        active = _close_active(active, idx, row, tf_mins)
                        active["cierre_tipo"] = "BOS_CONTRARIO"
                        rows.append(active)
                    active = None

                elif active["direccion"] == "SHORT" and bos == "BOS_BULL":
                    if active["duracion_barras"] >= MIN_BARS:
                        active = _close_active(active, idx, row, tf_mins)
                        active["cierre_tipo"] = "BOS_CONTRARIO"
                        rows.append(active)
                    active = None

        # --- Criterio 1: CHoCH confirma giro y cierra + abre nuevo ---
        if ce == "CHOCH_ALCISTA":
            if active and active["direccion"] == "SHORT":
                if active["duracion_barras"] >= MIN_BARS:
                    active = _close_active(active, idx, row, tf_mins)
                    active["cierre_tipo"] = "CHOCH"
                    rows.append(active)
                active = None
            # Abrir nuevo LONG solo si no hay uno ya activo en la misma direccion
            if not active or active["direccion"] != "LONG":
                mov_id += 1
                active = {
                    "mov_id": f"{tf_name}_{mov_id}", "tf": tf_name,
                    "direccion": "LONG", "timestamp_inicio": idx,
                    "timestamp_fin": None, "precio_inicio": close,
                    "precio_fin": np.nan,
                    "precio_extremo": row.get("high", close),
                    "magnitud_pct": 0.0, "duracion_barras": 1,
                    "duracion_min": 0, "tf_parent_id": None,
                    "cierre_tipo": None
                }

        elif ce == "CHOCH_BAJISTA":
            if active and active["direccion"] == "LONG":
                if active["duracion_barras"] >= MIN_BARS:
                    active = _close_active(active, idx, row, tf_mins)
                    active["cierre_tipo"] = "CHOCH"
                    rows.append(active)
                active = None
            if not active or active["direccion"] != "SHORT":
                mov_id += 1
                active = {
                    "mov_id": f"{tf_name}_{mov_id}", "tf": tf_name,
                    "direccion": "SHORT", "timestamp_inicio": idx,
                    "timestamp_fin": None, "precio_inicio": close,
                    "precio_fin": np.nan,
                    "precio_extremo": row.get("low", close),
                    "magnitud_pct": 0.0, "duracion_barras": 1,
                    "duracion_min": 0, "tf_parent_id": None,
                    "cierre_tipo": None
                }

        # --- Criterio 2: BOS sin CHoCH previo abre nuevo movimiento ---
        elif ce == "NONE" and not active:
            if bos == "BOS_BULL":
                mov_id += 1
                active = {
                    "mov_id": f"{tf_name}_{mov_id}", "tf": tf_name,
                    "direccion": "LONG", "timestamp_inicio": idx,
                    "timestamp_fin": None, "precio_inicio": close,
                    "precio_fin": np.nan,
                    "precio_extremo": row.get("high", close),
                    "magnitud_pct": 0.0, "duracion_barras": 1,
                    "duracion_min": 0, "tf_parent_id": None,
                    "cierre_tipo": None
                }
            elif bos == "BOS_BEAR":
                mov_id += 1
                active = {
                    "mov_id": f"{tf_name}_{mov_id}", "tf": tf_name,
                    "direccion": "SHORT", "timestamp_inicio": idx,
                    "timestamp_fin": None, "precio_inicio": close,
                    "precio_fin": np.nan,
                    "precio_extremo": row.get("low", close),
                    "magnitud_pct": 0.0, "duracion_barras": 1,
                    "duracion_min": 0, "tf_parent_id": None,
                    "cierre_tipo": None
                }

    # Ultimo movimiento abierto: cerrar en la ultima barra disponible
    if active and active["duracion_barras"] >= MIN_BARS:
        last_idx = df_struct.index[-1]
        last_row = df_struct.iloc[-1]
        active = _close_active(active, last_idx, last_row, tf_mins)
        active["cierre_tipo"] = "FIN_DATOS"
        rows.append(active)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _link_movements(all_movs: dict) -> pd.DataFrame:
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


def run_movements(symbol: str, structure_data: dict) -> pd.DataFrame:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    all_movs = {}
    for tf_name, df_struct in structure_data.items():
        print(f"  [{tf_name}] Extrayendo movimientos...")
        df_mov = _extract_movements(df_struct, tf_name)
        if not df_mov.empty:
            df_mov.to_csv(
                os.path.join(PROCESSED_DIR, f"{symbol}_movements_{tf_name}.csv"),
                index=False)
            print(f"  [{tf_name}] {len(df_mov)} movimientos detectados")
        else:
            print(f"  [{tf_name}] 0 movimientos detectados")
        all_movs[tf_name] = df_mov

    linked = _link_movements(all_movs)
    if not linked.empty:
        linked.to_csv(
            os.path.join(PROCESSED_DIR, f"{symbol}_movements_linked.csv"),
            index=False)
        print(f"  Total vinculados: {len(linked)} movimientos")
    return linked
