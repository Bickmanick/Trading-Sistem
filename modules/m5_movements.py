"""
MODULO 5 - DETECTOR DE MOVIMIENTOS

Fixes v2:
  - Bug precio_extremo: en la barra que dispara AGOTAMIENTO, el extremo
    se actualizaba ANTES de chequear el retroceso. Si esa barra retrocedia
    fuertemente, sobreescribia el extremo real con un valor cercano al
    precio_fin -> rango casi 0 -> fib_agotamiento siempre 1.0.
    Fix: guardar el extremo ANTES de actualizar en la barra actual, y
    usar ese extremo previo tanto para chequear retroceso como para cerrar.
  - Bug precio_inicio: se usaba close de la barra del evento. El movimiento
    empieza realmente en el open de esa misma barra (el evento ocurre
    al cierre, pero el precio de entrada es el open de la barra siguiente).
    Fix: precio_inicio = open de la barra siguiente (next_open), con
    fallback a close si no hay siguiente.
  - _link_movements vectorizado con merge_asof en vez de loop fila a fila.

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

EXHAUSTION_PCT = {
    "1M":  0.30, "5M":  0.50, "15M": 0.80, "30M": 1.00,
    "1H":  1.50, "4H":  2.50, "1D":  4.00, "1W":  6.00,
}
MIN_BARS = 3


def _close_active(active: dict, idx, row, tf_mins: int) -> dict:
    """Cierra el movimiento activo. precio_extremo ya debe estar fijado antes de llamar."""
    active["timestamp_fin"] = idx
    active["precio_fin"]    = row.get("close", np.nan)
    pi = active["precio_inicio"]
    pe = active["precio_extremo"]  # extremo fijado ANTES de la barra de cierre
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

    # Precalcular opens desplazados para precio_inicio correcto
    opens = df_struct["open"].values if "open" in df_struct.columns else None
    idx_list = list(df_struct.index)
    n_bars   = len(idx_list)

    for bar_i, (idx, row) in enumerate(df_struct.iterrows()):
        ce    = row.get("choch_event", "NONE")
        bos   = row.get("bos_event",   "NONE")
        close = row.get("close", np.nan)
        high  = row.get("high",  close)
        low   = row.get("low",   close)

        # precio de entrada = open de la SIGUIENTE barra (o close si es la ultima)
        if opens is not None and bar_i + 1 < n_bars:
            next_open = float(opens[bar_i + 1])
        else:
            next_open = float(close) if not pd.isna(close) else np.nan

        # --- Actualizar extremo del movimiento activo ---
        if active:
            # FIX: guardar extremo ANTES de actualizar con la barra actual
            pe_prev = active["precio_extremo"]

            if active["direccion"] == "LONG":
                if high > active["precio_extremo"]:
                    active["precio_extremo"] = high
            else:
                if low < active["precio_extremo"]:
                    active["precio_extremo"] = low
            active["duracion_barras"] += 1

            # Criterio 3: agotamiento — usar pe_prev (extremo antes de esta barra)
            pe = pe_prev  # extremo real alcanzado hasta la barra anterior
            if pe and pe > 0:
                if active["direccion"] == "LONG":
                    retroceso = (pe - close) / pe * 100
                else:
                    retroceso = (close - pe) / pe * 100
                if retroceso >= exh_pct and active["duracion_barras"] >= MIN_BARS:
                    # Restaurar extremo previo para que el calculo de fib sea correcto
                    active["precio_extremo"] = pe_prev
                    active = _close_active(active, idx, row, tf_mins)
                    active["cierre_tipo"] = "AGOTAMIENTO"
                    rows.append(active)
                    active = None
                    continue

            # Criterio 2: BOS contrario
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

        # --- Criterio 1: CHoCH ---
        if ce == "CHOCH_ALCISTA":
            if active and active["direccion"] == "SHORT":
                if active["duracion_barras"] >= MIN_BARS:
                    active = _close_active(active, idx, row, tf_mins)
                    active["cierre_tipo"] = "CHOCH"
                    rows.append(active)
                active = None
            if not active or active["direccion"] != "LONG":
                mov_id += 1
                active = {
                    "mov_id": f"{tf_name}_{mov_id}", "tf": tf_name,
                    "direccion": "LONG", "timestamp_inicio": idx,
                    "timestamp_fin": None,
                    "precio_inicio": next_open,  # FIX: open de barra siguiente
                    "precio_fin": np.nan,
                    "precio_extremo": high,
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
                    "timestamp_fin": None,
                    "precio_inicio": next_open,  # FIX: open de barra siguiente
                    "precio_fin": np.nan,
                    "precio_extremo": low,
                    "magnitud_pct": 0.0, "duracion_barras": 1,
                    "duracion_min": 0, "tf_parent_id": None,
                    "cierre_tipo": None
                }

        elif ce == "NONE" and not active:
            if bos == "BOS_BULL":
                mov_id += 1
                active = {
                    "mov_id": f"{tf_name}_{mov_id}", "tf": tf_name,
                    "direccion": "LONG", "timestamp_inicio": idx,
                    "timestamp_fin": None,
                    "precio_inicio": next_open,
                    "precio_fin": np.nan,
                    "precio_extremo": high,
                    "magnitud_pct": 0.0, "duracion_barras": 1,
                    "duracion_min": 0, "tf_parent_id": None,
                    "cierre_tipo": None
                }
            elif bos == "BOS_BEAR":
                mov_id += 1
                active = {
                    "mov_id": f"{tf_name}_{mov_id}", "tf": tf_name,
                    "direccion": "SHORT", "timestamp_inicio": idx,
                    "timestamp_fin": None,
                    "precio_inicio": next_open,
                    "precio_fin": np.nan,
                    "precio_extremo": low,
                    "magnitud_pct": 0.0, "duracion_barras": 1,
                    "duracion_min": 0, "tf_parent_id": None,
                    "cierre_tipo": None
                }

    if active and active["duracion_barras"] >= MIN_BARS:
        last_idx = df_struct.index[-1]
        last_row = df_struct.iloc[-1]
        active = _close_active(active, last_idx, last_row, tf_mins)
        active["cierre_tipo"] = "FIN_DATOS"
        rows.append(active)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _link_movements(all_movs: dict) -> pd.DataFrame:
    """Vincula movimientos con su padre de TF superior via merge_asof."""
    frames = [df.copy() for df in all_movs.values() if not df.empty]
    if not frames:
        return pd.DataFrame()
    linked = pd.concat(frames, ignore_index=True)
    linked["timestamp_inicio"] = pd.to_datetime(linked["timestamp_inicio"], utc=True)
    linked["timestamp_fin"]    = pd.to_datetime(linked["timestamp_fin"],    utc=True)
    linked["tf_parent_id"]     = None

    # Para cada TF, buscar padre en TF superior con merge_asof
    for i, tf in enumerate(TF_ORDER[:-1]):
        tf_up = TF_ORDER[i + 1]
        child  = linked[linked["tf"] == tf].sort_values("timestamp_inicio")
        parent = linked[linked["tf"] == tf_up][["mov_id", "timestamp_inicio",
                                                  "timestamp_fin"]].copy()
        parent = parent.rename(columns={"mov_id": "_parent_id",
                                         "timestamp_inicio": "_p_inicio",
                                         "timestamp_fin":    "_p_fin"})
        parent = parent.sort_values("_p_inicio")
        if child.empty or parent.empty:
            continue
        # merge_asof: para cada hijo encuentra el padre que empezo antes
        merged = pd.merge_asof(
            child[["mov_id", "timestamp_inicio"]],
            parent,
            left_on="timestamp_inicio",
            right_on="_p_inicio",
            direction="backward"
        )
        # Solo vincular si el padre NO ha terminado antes del inicio del hijo
        valid = merged[merged["_p_fin"].isna() |
                       (merged["_p_fin"] >= merged["timestamp_inicio"])]
        id_map = dict(zip(valid["mov_id"], valid["_parent_id"]))
        mask   = linked["mov_id"].isin(id_map)
        linked.loc[mask, "tf_parent_id"] = linked.loc[mask, "mov_id"].map(id_map)

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
