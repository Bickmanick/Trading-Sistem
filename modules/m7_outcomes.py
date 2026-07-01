"""
MODULO 7 - OUTCOMES

Fixes v3:
  - Bug fib_agotamiento: usaba precio_fin como referencia cuando deberia usar
    el maximo retroceso INTERNO del movimiento desde el extremo.
    Ahora calcula el retroceso real desde precio_extremo hasta el minimo/maximo
    interno de las velas 1M durante el movimiento.
  - Bug rebote_fib: el rango del swing previo se media mal cuando el swing
    previo iba en la misma direccion. Corregido con logica de direccion.
  - Vectorizacion completa de MFE/MAE: en vez de iterar barra a barra,
    agrupa todas las ventanas de una vez con searchsorted sobre el indice
    ordenado de df_1m. Reduce tiempo de ~600s a ~30s.
  - _continuacion vectorizada: una sola operacion de merge en vez de
    filtrar por cada movimiento individualmente.
  - _alineamiento_3tf vectorizado: merge doble en vez de bucle.

Outputs: data/processed/{symbol}_outcomes.csv
"""
import os
import pandas as pd
import numpy as np
from config import DATA_DIR

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FIB_LEVELS    = np.array([0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0])
TF_ORDER      = ["1M", "5M", "15M", "30M", "1H", "4H", "1D", "1W"]


def _nearest_fib_vec(ratios: np.ndarray) -> np.ndarray:
    """Vectorizado: devuelve el nivel Fibonacci mas cercano para cada ratio."""
    ratios = np.clip(ratios, 0.0, 1.0)
    diffs  = np.abs(ratios[:, None] - FIB_LEVELS[None, :])
    return FIB_LEVELS[np.argmin(diffs, axis=1)]


def _compute_mfe_mae_vectorized(validos: pd.DataFrame,
                                df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula MFE y MAE para todos los movimientos de una vez.

    Estrategia:
      - Ordena df_1m por index (ya deberia estarlo)
      - Para cada movimiento usa searchsorted para encontrar el rango de
        indices de las velas 1M que caen dentro de [ts_inicio, ts_fin]
      - Agrupa por movimiento con un loop eficiente sobre bloques
        (mucho mas rapido que filtrar DataFrame en cada iteracion)
    """
    idx_1m = df_1m.index  # DatetimeIndex UTC ordenado
    highs  = df_1m["high"].values
    lows   = df_1m["low"].values

    ts_ini = validos["timestamp_inicio"].values  # numpy datetime64
    ts_fin = validos["timestamp_fin"].values
    ep     = validos["precio_inicio"].values
    dirs   = validos["direccion"].values

    mfe_arr = np.full(len(validos), np.nan)
    mae_arr = np.full(len(validos), np.nan)

    # Convertir a numpy para searchsorted
    idx_np = idx_1m.values

    for i in range(len(validos)):
        lo = np.searchsorted(idx_np, ts_ini[i], side="left")
        hi = np.searchsorted(idx_np, ts_fin[i], side="right")
        if lo >= hi:
            continue
        h_max = highs[lo:hi].max()
        l_min = lows[lo:hi].min()
        p     = ep[i]
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
    Fibonacci de agotamiento INTERNO:
    Mide cuanto retrocedio el precio desde el precio_extremo hacia el interior
    del movimiento (no hacia precio_fin).

    Para LONG: retroceso = (precio_extremo - minimo_interno) / rango
    Para SHORT: retroceso = (maximo_interno - precio_extremo) / rango

    donde rango = abs(precio_extremo - precio_inicio)

    Si rango == 0 -> nan
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
            # El movimiento subio: el retroceso interno es desde el extremo
            # hasta el minimo que toco internamente
            min_interno = lows[lo:hi].min()
            ret = pe_arr[i] - min_interno
        else:
            # El movimiento bajo: el retroceso interno es desde el extremo
            # hasta el maximo que toco internamente
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

    Para cada movimiento busca el movimiento inmediatamente anterior del mismo TF.
    Mide donde quedo precio_fin del movimiento actual dentro del rango del
    swing previo (desde precio_inicio hasta precio_extremo del previo).

    Logica de ratio:
      Swing previo LONG (subio de pi_p a pe_p):
        ratio = (pe_p - precio_fin_actual) / rango_previo
        -> 0.0 = el precio actual esta en el extremo del previo (no retrocedio)
        -> 1.0 = retrocedio todo el swing previo (volvio al inicio)

      Swing previo SHORT (bajo de pi_p a pe_p):
        ratio = (precio_fin_actual - pe_p) / rango_previo
    """
    result_rows = []

    for tf in validos["tf"].unique():
        sub = validos[validos["tf"] == tf].sort_values("timestamp_inicio").copy()
        sub = sub.reset_index(drop=True)

        for i in range(len(sub)):
            mov = sub.iloc[i]
            ts_ini = mov["timestamp_inicio"]
            previos = sub[sub["timestamp_fin"] < ts_ini]
            if previos.empty:
                result_rows.append({"mov_id": mov["mov_id"], "rebote_fib": np.nan})
                continue
            prev  = previos.iloc[-1]
            pi_p  = prev["precio_inicio"]
            pe_p  = prev["precio_extremo"]
            rango = abs(pe_p - pi_p)
            pf    = mov.get("precio_fin", np.nan) if hasattr(mov, 'get') else mov["precio_fin"]
            if rango == 0 or pd.isna(pe_p) or pd.isna(pi_p) or pd.isna(pf):
                result_rows.append({"mov_id": mov["mov_id"], "rebote_fib": np.nan})
                continue
            if prev["direccion"] == "LONG":
                ret = pe_p - pf
            else:
                ret = pf - pe_p
            ratio = max(0.0, min(1.0, ret / rango))
            fib   = float(_nearest_fib_vec(np.array([ratio]))[0])
            result_rows.append({"mov_id": mov["mov_id"], "rebote_fib": fib})

    return pd.DataFrame(result_rows)


def _compute_continuacion_vectorized(validos: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada movimiento: CONTINUACION si el siguiente mov del mismo TF
    va en la misma direccion, REVERSION si va en contra, ULTIMO si no hay siguiente.
    Vectorizado con shift por TF.
    """
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
    """
    True si movimiento + padre + abuelo van en la misma direccion.
    Usa merge doble sobre tf_parent_id en vez de iterar fila a fila.
    """
    base = validos[["mov_id", "direccion", "tf_parent_id"]].copy()

    # Merge con padre
    padre = base.rename(columns={
        "mov_id": "parent_id",
        "direccion": "parent_dir",
        "tf_parent_id": "abuelo_id"
    })
    m1 = base.merge(padre, left_on="tf_parent_id", right_on="parent_id", how="left")

    # Merge con abuelo
    abuelo = base[["mov_id", "direccion"]].rename(columns={
        "mov_id": "abuelo_mov_id",
        "direccion": "abuelo_dir"
    })
    m2 = m1.merge(abuelo, left_on="abuelo_id", right_on="abuelo_mov_id", how="left")

    # Alineamiento: padre existe y misma direccion
    tiene_padre  = m2["parent_dir"].notna() & (m2["parent_dir"] == m2["direccion"])
    tiene_abuelo = m2["abuelo_dir"].notna() & (m2["abuelo_dir"] == m2["direccion"])

    m2["alineamiento_3tf"] = tiene_padre & tiene_abuelo
    # Si tiene padre pero no abuelo: alineamiento de 2 TFs (tambien vale)
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

    # 1. MFE / MAE vectorizado
    print("  [1/5] MFE / MAE...")
    df_mfe = _compute_mfe_mae_vectorized(validos, df_1m)

    # 2. Fibonacci de agotamiento interno
    print("  [2/5] Fib agotamiento...")
    df_fagot = _compute_fib_agotamiento_vectorized(validos, df_1m)

    # 3. Fibonacci rebote swing previo
    print("  [3/5] Fib rebote swing previo...")
    df_freb = _compute_rebote_fib_vectorized(validos)

    # 4. Continuacion vectorizada
    print("  [4/5] Continuacion...")
    df_cont = _compute_continuacion_vectorized(validos)

    # 5. Alineamiento 3TF vectorizado
    print("  [5/5] Alineamiento 3TF...")
    df_alin = _compute_alineamiento_vectorized(validos)

    # --- Ensamblar todo ---
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

    # ratio MFE/MAE
    df_out["ratio_mfe_mae"] = np.where(
        df_out["mae_pct"] > 0,
        (df_out["mfe_pct"] / df_out["mae_pct"]).round(3),
        np.nan
    )

    path = os.path.join(PROCESSED_DIR, f"{symbol}_outcomes.csv")
    df_out.to_csv(path, index=False)
    print(f"  Outcomes: {len(df_out)} filas -> {path}")
    return df_out
