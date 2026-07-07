"""
MODULO 10 — WALK-FORWARD EXTENDIDO

Valida la robustez de los patrones con ventanas rodantes temporales.

Logica:
  - Divide el historico en N ventanas solapadas (rolling).
  - En cada ventana calcula el winrate de cada patron validado.
  - Un patron es ROBUSTO si:
      * En >= 70% de las ventanas su WR supera EDGE_WINRATE.
      * La desviacion estandar del WR entre ventanas es < MAX_WR_STD.
  - Añade columnas de robustez al CSV de patrones validados.

Output:
    data/results/{symbol}_patterns_robust.csv
"""
import os
import numpy as np
import pandas as pd
from config import DATA_DIR, EDGE_WINRATE

RESULTS_DIR = os.path.join(DATA_DIR, "results")
N_WINDOWS   = 5       # numero de ventanas rodantes
WINDOW_OVERLAP = 0.5  # solapamiento entre ventanas
MAX_WR_STD  = 0.15    # desviacion maxima aceptable del WR entre ventanas
MIN_ROBUST  = 0.70    # fraccion de ventanas donde el patron debe ser positivo


def _split_windows(df: pd.DataFrame, n: int, overlap: float) -> list:
    """
    Genera N ventanas (train, valid) con solapamiento.
    Cada ventana usa ~60% para train y ~40% para valid.
    """
    ts = df["timestamp_inicio"].sort_values()
    total = len(ts)
    step  = int(total * (1 - overlap) / max(n - 1, 1))
    win_size = int(total * 0.6)
    windows  = []
    for i in range(n):
        start = i * step
        mid   = start + win_size
        end   = mid + int(total * 0.4)
        if end > total:
            end = total
        if mid >= total:
            break
        t_train = (ts.iloc[start], ts.iloc[mid - 1])
        t_valid = (ts.iloc[mid],   ts.iloc[end - 1])
        windows.append((t_train, t_valid))
    return windows


def _wr_in_window(df: pd.DataFrame, col: str, val: str,
                  tipo: str, direction: str,
                  t_start, t_end) -> float:
    sub = df[(df["timestamp_inicio"] >= t_start) &
             (df["timestamp_inicio"] <= t_end) &
             (df["direccion"] == direction)]
    if tipo == "bool":
        sub = sub[sub[col].astype(float) == float(val)]
    else:
        sub = sub[sub[col].astype(str) == str(val)]
    if len(sub) < 3:
        return np.nan
    return (sub["continuacion"] == "CONTINUACION").mean()


def run_walkforward(symbol: str,
                    df_start: pd.DataFrame,
                    df_outcomes: pd.DataFrame,
                    validated: pd.DataFrame) -> pd.DataFrame:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if validated is None or validated.empty:
        print("  M10: sin patrones validados. Skip.")
        return pd.DataFrame()

    # Unir dataset base
    keep_out = ["mov_id", "continuacion", "rebote_fib", "fib_agotamiento",
                "mfe_pct", "mae_pct", "cierre_tipo", "alineamiento_3tf"]
    keep_out = [c for c in keep_out if c in df_outcomes.columns]
    df_out   = df_outcomes[keep_out].copy()
    df_out   = df_out[df_out["continuacion"] != "ULTIMO"]
    df_joined = df_start.merge(df_out, on="mov_id", how="inner")
    df_joined["timestamp_inicio"] = pd.to_datetime(
        df_joined["timestamp_inicio"], utc=True)

    windows = _split_windows(df_joined, N_WINDOWS, WINDOW_OVERLAP)
    print(f"  M10: {len(windows)} ventanas rodantes sobre "
          f"{len(df_joined)} movimientos")

    robust_rows = []
    for _, p in validated.iterrows():
        col  = p.get("variables")
        raw  = str(p.get("patron", ""))
        if "==" not in raw or col not in df_joined.columns:
            continue
        val   = raw.split("==")[1].split("@")[0]
        tipo  = p.get("tipo", "cat")
        dir_  = p.get("direccion")
        tf_p  = p.get("tf")

        base = df_joined[df_joined["tf"] == tf_p] if pd.notna(tf_p) else df_joined

        wrs = []
        for (t_tr, _), (t_v0, t_v1) in windows:
            wr = _wr_in_window(base, col, val, tipo, dir_, t_v0, t_v1)
            if not np.isnan(wr):
                wrs.append(wr)

        if not wrs:
            continue

        wrs_arr  = np.array(wrs)
        pct_edge = (wrs_arr >= EDGE_WINRATE).mean()
        wr_std   = wrs_arr.std()
        robusto  = bool(pct_edge >= MIN_ROBUST and wr_std <= MAX_WR_STD)

        row = p.to_dict()
        row["wr_ventanas"]   = np.round(wrs_arr, 4).tolist()
        row["wr_std"]        = round(wr_std, 4)
        row["pct_ventanas_edge"] = round(pct_edge, 4)
        row["robusto"]       = robusto
        robust_rows.append(row)

    if not robust_rows:
        print("  M10: ningun patron supero validacion de robustez.")
        return pd.DataFrame()

    df_robust = pd.DataFrame(robust_rows)
    n_ok = df_robust["robusto"].sum()
    print(f"  M10: {n_ok}/{len(df_robust)} patrones ROBUSTOS")

    path = os.path.join(RESULTS_DIR, f"{symbol}_patterns_robust.csv")
    df_robust.to_csv(path, index=False)
    print(f"  M10: guardado -> {path}")
    return df_robust
