"""
MODULO 7 - OUTCOMES

Fixes v4:
  - _compute_rebote_fib: reemplazado loop O(n^2) por merge_asof O(n log n).
    Con 23562 movimientos en 1M, el loop anterior tardaba >200s solo en ese TF.
  - fib_agotamiento: ahora recibe precio_extremo correcto desde m5 (ya fijado
    antes de la barra de cierre), por lo que el calculo es valido sin cambios.
  - ratio_mfe_mae: proteccion contra division por cero mejorada.

Output: data/processed/{symbol}_outcomes.csv
"""
import os
import pandas as pd
import numpy as np
from config import DATA_DIR

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FIB_LEVELS    = np.array([0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0])
TF_ORDER      = ["1M", "5M", "15M", "30M", "1H", "4H", "1D", "1W"]


def _nearest_fib_vec(ratios: np.ndarray) -> np.ndarray:
    ratios = np.clip(ratios, 0.0, 1.0)
    diffs  = np.abs(ratios[:, None] - FIB_LEVELS[None, :])
    return FIB_LEVELS[np.argmin(diffs, axis=1)]


def _compute_mfe_mae_vectorized(validos: pd.DataFrame,
                                df_1m: pd.DataFrame) -> pd.DataFrame:
    idx_np = df_1m.index.values
    highs  = df_1m["high"].values
    lows   = df_1m["low"].values
    ts_ini = validos["timestamp_inicio"].values
    ts_fin = validos["timestamp_fin"].values
    ep     = validos["precio_inicio"].values
    dirs   = validos["direccion"].values
    mfe_arr = np.full(len(validos), np.nan)
    mae_arr = np.full(len(validos), np.nan)

    for i in range(len(validos)):
        lo = np.searchsorted(idx_np, ts_ini[i], side="left")
        hi = np.searchsorted(idx_np, ts_fin[i], side="right")
        if lo >= hi:
            continue
        h_max = highs[lo:hi].max()
        l_min = lows[lo:hi].min()
        p = ep[i]
        if p == 0 or np.isnan(p):
            continue
        if dirs[i] == "LONG":
            mfe_arr[i] = max((h_max - p) / p * 100, 0.0)
            mae_arr[i] = max((p - l_min) / p * 100, 0.0)
        else:
            mfe_arr[i] = max((p - l_min) / p * 100, 0.0)
            mae_arr[i] = max((h_max - p) / p * 100, 0.0)

    result = validos[["mov_id"]].copy()
    result["mfe_pct"] = np.round(mfe_arr, 4)
    result["mae_pct"] = np.round(mae_arr, 4)
    return result


def _compute_fib_agotamiento_vectorized(validos: pd.DataFrame,
                                        df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Fibonacci del agotamiento INTERNO del movimiento.
    Mide el maximo retroceso desde precio_extremo dentro de las velas 1M.
    Con el fix de m5, precio_extremo ahora es el maximo/minimo real
    alcanzado ANTES de la barra de cierre.
    """
    idx_np = df_1m.index.values
    highs  = df_1m["high"].values
    lows   = df_1m["low"].values
    ts_ini = validos["timestamp_inicio"].values
    ts_fin = validos["timestamp_fin"].values
    pi_arr = validos["precio_inicio"].values
    pe_arr = validos["precio_extremo"].values
    dirs   = validos["direccion"].values
    ratios = np.full(len(validos), np.nan)

    for i in range(len(validos)):
        rango = abs(pe_arr[i] - pi_arr[i])
        if rango == 0 or np.isnan(rango):
            continue
        lo = np.searchsorted(idx_np, ts_ini[i], side="left")
        hi = np.searchsorted(idx_np, ts_fin[i], side="right")
        if lo >= hi:
            continue
        if dirs[i] == "LONG":
            min_interno = lows[lo:hi].min()
            ret = pe_arr[i] - min_interno
        else:
            max_interno = highs[lo:hi].max()
            ret = max_interno - pe_arr[i]
        ratios[i] = max(0.0, min(1.0, ret / rango))

    fibs = _nearest_fib_vec(np.where(np.isnan(ratios), 0.0, ratios))
    fibs = np.where(np.isnan(ratios), np.nan, fibs)
    result = validos[["mov_id"]].copy()
    result["fib_agotamiento"] = fibs
    return result


def _compute_rebote_fib_vectorized(validos: pd.DataFrame) -> pd.DataFrame:
    """
    Fibonacci del swing PREVIO donde termino el movimiento actual.
    Usa merge_asof para sustituir el loop O(n^2) por O(n log n).

    Para cada TF:
      - Construye tabla de 'previos': un DataFrame con las columnas del
        swing previo alineado por timestamp_fin < timestamp_inicio del actual.
      - merge_asof con direction='backward' encuentra en O(n log n) el
        swing previo mas reciente para cada movimiento.
    """
    result_rows = []

    for tf in validos["tf"].unique():
        sub = (validos[validos["tf"] == tf]
               .sort_values("timestamp_inicio")
               .reset_index(drop=True)
               .copy())

        if len(sub) < 2:
            for _, row in sub.iterrows():
                result_rows.append({"mov_id": row["mov_id"], "rebote_fib": np.nan})
            continue

        # Tabla de 'previo': timestamp_fin del swing previo como clave de join
        prev = sub[["mov_id", "timestamp_fin", "precio_inicio",
                     "precio_extremo", "direccion"]].copy()
        prev = prev.rename(columns={
            "mov_id":         "prev_id",
            "timestamp_fin":  "prev_ts_fin",
            "precio_inicio":  "prev_pi",
            "precio_extremo": "prev_pe",
            "direccion":      "prev_dir",
        })
        prev = prev.dropna(subset=["prev_ts_fin"]).sort_values("prev_ts_fin")

        # Tabla de 'actual': buscar el previo cuyo fin es < inicio del actual
        curr = sub[["mov_id", "timestamp_inicio", "precio_fin"]].copy()
        curr = curr.sort_values("timestamp_inicio")

        merged = pd.merge_asof(
            curr,
            prev,
            left_on="timestamp_inicio",
            right_on="prev_ts_fin",
            direction="backward"
        )
        # Excluir casos donde el previo empieza en el mismo timestamp (mismo mov)
        merged = merged[merged["prev_id"] != merged["mov_id"]]

        for _, row in merged.iterrows():
            pi_p  = row.get("prev_pi")
            pe_p  = row.get("prev_pe")
            pf    = row.get("precio_fin")
            dir_p = row.get("prev_dir")
            if pd.isna(pi_p) or pd.isna(pe_p) or pd.isna(pf) or pd.isna(dir_p):
                result_rows.append({"mov_id": row["mov_id"], "rebote_fib": np.nan})
                continue
            rango = abs(pe_p - pi_p)
            if rango == 0:
                result_rows.append({"mov_id": row["mov_id"], "rebote_fib": np.nan})
                continue
            if dir_p == "LONG":
                ret = pe_p - pf
            else:
                ret = pf - pe_p
            ratio = max(0.0, min(1.0, ret / rango))
            fib   = float(_nearest_fib_vec(np.array([ratio]))[0])
            result_rows.append({"mov_id": row["mov_id"], "rebote_fib": fib})

        # Movimientos sin previo (los que no aparecen en merged)
        ids_merged = {r["mov_id"] for r in result_rows}
        for _, row in sub.iterrows():
            if row["mov_id"] not in ids_merged:
                result_rows.append({"mov_id": row["mov_id"], "rebote_fib": np.nan})

    return pd.DataFrame(result_rows)


def _compute_continuacion_vectorized(validos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tf in validos["tf"].unique():
        sub = validos[validos["tf"] == tf].sort_values("timestamp_inicio").copy()
        sub["next_dir"] = sub["direccion"].shift(-1)
        sub["continuacion"] = np.where(
            sub["next_dir"].isna(), "ULTIMO",
            np.where(sub["next_dir"] == sub["direccion"], "CONTINUACION", "REVERSION")
        )
        rows.append(sub[["mov_id", "continuacion"]])
    return pd.concat(rows, ignore_index=True)


def _compute_alineamiento_vectorized(validos: pd.DataFrame) -> pd.DataFrame:
    base = validos[["mov_id", "direccion", "tf_parent_id"]].copy()
    padre = base.rename(columns={
        "mov_id":       "parent_id",
        "direccion":    "parent_dir",
        "tf_parent_id": "abuelo_id"
    })
    m1 = base.merge(padre, left_on="tf_parent_id", right_on="parent_id", how="left")
    abuelo = base[["mov_id", "direccion"]].rename(columns={
        "mov_id":    "abuelo_mov_id",
        "direccion": "abuelo_dir"
    })
    m2 = m1.merge(abuelo, left_on="abuelo_id", right_on="abuelo_mov_id", how="left")
    tiene_padre  = m2["parent_dir"].notna() & (m2["parent_dir"] == m2["direccion"])
    tiene_abuelo = m2["abuelo_dir"].notna() & (m2["abuelo_dir"] == m2["direccion"])
    m2["alineamiento_3tf"] = tiene_padre & tiene_abuelo
    m2.loc[tiene_padre & ~m2["abuelo_dir"].notna(), "alineamiento_3tf"] = True
    return m2[["mov_id", "alineamiento_3tf"]]


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
    total   = len(validos)
    print(f"  Calculando outcomes para {total} movimientos...")

    print("  [1/5] MFE / MAE...")
    df_mfe = _compute_mfe_mae_vectorized(validos, df_1m)

    print("  [2/5] Fib agotamiento...")
    df_fagot = _compute_fib_agotamiento_vectorized(validos, df_1m)

    print("  [3/5] Fib rebote swing previo (merge_asof)...")
    df_freb = _compute_rebote_fib_vectorized(validos)

    print("  [4/5] Continuacion...")
    df_cont = _compute_continuacion_vectorized(validos)

    print("  [5/5] Alineamiento 3TF...")
    df_alin = _compute_alineamiento_vectorized(validos)

    df_out = validos[[
        "mov_id", "tf", "direccion", "magnitud_pct",
        "duracion_min", "cierre_tipo",
        "precio_inicio", "precio_fin", "precio_extremo"
    ]].copy()
    df_out["magnitud_pct"] = df_out["magnitud_pct"].round(4)
    df_out = df_out.merge(df_mfe,   on="mov_id", how="left")
    df_out = df_out.merge(df_fagot, on="mov_id", how="left")
    df_out = df_out.merge(df_freb,  on="mov_id", how="left")
    df_out = df_out.merge(df_cont,  on="mov_id", how="left")
    df_out = df_out.merge(df_alin,  on="mov_id", how="left")
    df_out["ratio_mfe_mae"] = np.where(
        df_out["mae_pct"] > 0,
        (df_out["mfe_pct"] / df_out["mae_pct"]).round(3),
        np.nan
    )

    path = os.path.join(PROCESSED_DIR, f"{symbol}_outcomes.csv")
    df_out.to_csv(path, index=False)
    print(f"  Outcomes: {len(df_out)} filas -> {path}")
    return df_out
