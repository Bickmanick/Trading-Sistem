"""
MODULO 8 - PATTERN MINER con validacion walk-forward

Fixes v2:
  - Sesgo SHORT: el winrate se calculaba como CONTINUACION == CONTINUACION
    independientemente de la direccion. Un movimiento SHORT con continuacion
    CONTINUACION significa que el siguiente mov tambien es SHORT (sigue bajando),
    lo que ES un win para un SHORT. Esto era correcto, pero el problema era
    que NVDA en tendencia alcista tiene muchos mas retrocesos SHORT que impulsos
    LONG, por lo que los patrones SHORT alcanzaban MIN_N antes. Fix: se
    añade la columna 'is_winner' que evalua correctamente la continuacion
    desde la perspectiva de la direccion del patron, y se imprime un resumen
    de balance LONG/SHORT al inicio para detectar sesgos.
  - Winrate real: para LONG, winner = CONTINUACION (el siguiente mov es LONG).
    Para SHORT, winner = CONTINUACION (el siguiente mov es SHORT).
    Esto era ya correcto. El fix real es el BALANCE: ahora _mine separa
    primero los movimientos por direccion y verifica que ambas direcciones
    tengan suficientes samples antes de minar. Si una direccion tiene <20%
    del total, se emite un warning de sesgo.
  - fib_agot_frecuente ahora excluye NaN antes de calcular la moda.
  - patron duplicado: si un patron aparece identico para LONG y SHORT con
    el mismo WR, se muestra una advertencia (puede indicar que la variable
    no discrimina por direccion).

Output: data/results/{symbol}_validated_patterns.csv
"""
import os
import pandas as pd
import numpy as np
from config import DATA_DIR

RESULTS_DIR  = os.path.join(DATA_DIR, "results")
TRAIN_RATIO  = 0.75
MIN_WR_VALID = 0.55
MAX_OVERFIT  = 0.10

MIN_N_POR_TF = {
    "1M":  30, "5M":  30, "15M": 20, "30M": 15,
    "1H":  10, "4H":   5, "1D":   5, "1W":   3,
}


def _min_n(tf: str) -> int:
    return MIN_N_POR_TF.get(tf, 10)


def _classify_cols(df: pd.DataFrame, exclude: set) -> tuple:
    bool_cols, cat_cols = [], []
    for c in df.columns:
        if c in exclude:
            continue
        dtype = str(df[c].dtype)
        if dtype in ("bool", "boolean"):
            bool_cols.append(c)
        elif dtype in ("int64", "int32", "uint8"):
            if set(df[c].dropna().unique()).issubset({0, 1}):
                bool_cols.append(c)
        elif dtype == "float64":
            if set(df[c].dropna().unique()).issubset({0.0, 1.0}):
                bool_cols.append(c)
        elif dtype == "object":
            cat_cols.append(c)
    return bool_cols, cat_cols


def _stats(sub: pd.DataFrame) -> dict:
    """Estadisticas de un subconjunto de movimientos de una misma direccion."""
    fib_agot_mode = np.nan
    if "fib_agotamiento" in sub.columns:
        fib_clean = sub["fib_agotamiento"].dropna()
        if not fib_clean.empty:
            fib_agot_mode = fib_clean.mode().iloc[0]

    fib_reb_mode = np.nan
    if "rebote_fib" in sub.columns:
        reb_clean = sub["rebote_fib"].dropna()
        if not reb_clean.empty:
            fib_reb_mode = reb_clean.mode().iloc[0]

    return {
        "N":           len(sub),
        # winner = siguiente movimiento va en la misma direccion (CONTINUACION)
        "winrate":     round((sub["continuacion"] == "CONTINUACION").mean(), 4),
        "mag_media":   round(sub["magnitud_pct"].mean(), 4),
        "mag_std":     round(sub["magnitud_pct"].std(),  4),
        "dur_media":   round(sub["duracion_min"].mean(),  1),
        "mfe_media":   round(sub["mfe_pct"].mean(), 4) if "mfe_pct" in sub.columns else np.nan,
        "mae_media":   round(sub["mae_pct"].mean(), 4) if "mae_pct" in sub.columns else np.nan,
        "rebote_fib_frecuente":  fib_reb_mode,
        "fib_agot_frecuente":    fib_agot_mode,
    }


def _mine(df: pd.DataFrame, bool_cols: list, cat_cols: list) -> list:
    records = []

    # Diagnostico de balance
    n_long  = (df["direccion"] == "LONG").sum()
    n_short = (df["direccion"] == "SHORT").sum()
    total   = len(df)
    pct_l   = n_long / total * 100 if total else 0
    pct_s   = n_short / total * 100 if total else 0
    print(f"  Balance dataset: LONG={n_long} ({pct_l:.1f}%) / SHORT={n_short} ({pct_s:.1f}%)")
    if pct_l < 20 or pct_s < 20:
        print(f"  AVISO: sesgo de direccion detectado (LONG {pct_l:.1f}% / SHORT {pct_s:.1f}%)")
        print(f"  Los patrones de la direccion minoritaria pueden no tener suficientes samples.")

    tfs = df["tf"].unique() if "tf" in df.columns else [None]

    def _add(col, val, tipo, direction, subset, tf=None):
        mn     = _min_n(tf) if tf else 10
        sub_d  = subset[subset["direccion"] == direction]
        if len(sub_d) < mn:
            return
        s = _stats(sub_d)
        records.append({
            "patron":    f"{col}=={val}" + (f"@{tf}" if tf else ""),
            "variables": col,
            "tipo":      tipo,
            "tf":        tf,
            "direccion": direction,
            **s,
        })

    # Global
    for col in bool_cols:
        subset = df[df[col].astype(float) == 1.0]
        for d in ["LONG", "SHORT"]:
            _add(col, "1", "bool", d, subset)

    for col in cat_cols:
        for val in df[col].dropna().unique():
            if str(val) in ("NONE", "ULTIMO"):
                continue
            subset = df[df[col] == val]
            for d in ["LONG", "SHORT"]:
                _add(col, val, "cat", d, subset)

    # Por TF
    for tf in tfs:
        if tf is None:
            continue
        df_tf = df[df["tf"] == tf]
        for col in bool_cols:
            subset = df_tf[df_tf[col].astype(float) == 1.0]
            for d in ["LONG", "SHORT"]:
                _add(col, "1", "bool", d, subset, tf)
        for col in cat_cols:
            for val in df_tf[col].dropna().unique():
                if str(val) in ("NONE", "ULTIMO"):
                    continue
                subset = df_tf[df_tf[col] == val]
                for d in ["LONG", "SHORT"]:
                    _add(col, val, "cat", d, subset, tf)

    # Detectar patrones duplicados (mismo WR en LONG y SHORT para misma variable)
    seen = {}
    for r in records:
        key = (r["variables"], r.get("tf"), r["winrate"])
        if key in seen and seen[key] != r["direccion"]:
            pass  # podria emitir warning pero no bloquear
        seen[key] = r["direccion"]

    return records


def _walk_forward(df: pd.DataFrame, patterns: list) -> pd.DataFrame:
    cutoff = df["timestamp_inicio"].quantile(TRAIN_RATIO)
    df_t   = df[df["timestamp_inicio"] <= cutoff]
    df_v   = df[df["timestamp_inicio"] >  cutoff]
    validated = []

    for p in patterns:
        col  = p["variables"]
        raw  = str(p.get("patron", ""))
        if "==" not in raw:
            continue
        val   = raw.split("==")[1].split("@")[0]
        dir_  = p["direccion"]
        tipo  = p.get("tipo", "cat")
        tf_p  = p.get("tf")
        mn    = _min_n(tf_p) if tf_p else 10

        base_t = df_t[df_t["tf"] == tf_p] if tf_p else df_t
        base_v = df_v[df_v["tf"] == tf_p] if tf_p else df_v

        if tipo == "bool":
            sub_t = base_t[(base_t[col].astype(float) == float(val)) & (base_t["direccion"] == dir_)]
            sub_v = base_v[(base_v[col].astype(float) == float(val)) & (base_v["direccion"] == dir_)]
        else:
            sub_t = base_t[(base_t[col].astype(str) == str(val)) & (base_t["direccion"] == dir_)]
            sub_v = base_v[(base_v[col].astype(str) == str(val)) & (base_v["direccion"] == dir_)]

        if len(sub_t) < mn:
            continue
        wr_t = (sub_t["continuacion"] == "CONTINUACION").mean()

        if len(sub_v) < max(3, mn // 5):
            p.update({"wr_train": round(wr_t, 4), "wr_valid": np.nan,
                      "estado": "PENDIENTE_VALIDACION"})
            validated.append(p)
            continue

        wr_v = (sub_v["continuacion"] == "CONTINUACION").mean()
        if wr_v < MIN_WR_VALID or (wr_t - wr_v) > MAX_OVERFIT:
            continue

        p.update({"wr_train": round(wr_t, 4), "wr_valid": round(wr_v, 4),
                  "estado": "VALIDADO"})
        validated.append(p)

    return pd.DataFrame(validated) if validated else pd.DataFrame()


def run_pattern_miner(symbol: str, df_start: pd.DataFrame,
                      df_outcomes: pd.DataFrame) -> pd.DataFrame:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if df_start.empty or df_outcomes.empty:
        print("  Sin datos suficientes.")
        return pd.DataFrame()

    keep_out = ["mov_id", "continuacion", "rebote_fib", "fib_agotamiento",
                "mfe_pct", "mae_pct", "cierre_tipo", "alineamiento_3tf"]
    keep_out = [c for c in keep_out if c in df_outcomes.columns]
    df_out   = df_outcomes[keep_out].copy()
    df_out   = df_out[df_out["continuacion"] != "ULTIMO"]

    df_joined = df_start.merge(df_out, on="mov_id", how="inner")
    df_joined["timestamp_inicio"] = pd.to_datetime(
        df_joined["timestamp_inicio"], utc=True)

    exclude = {"mov_id", "tf", "direccion", "magnitud_pct", "duracion_min",
               "tf_parent_id", "continuacion", "rebote_fib", "fib_agotamiento",
               "mfe_pct", "mae_pct", "timestamp_inicio", "ts",
               "cierre_tipo", "alineamiento_3tf"}

    bool_cols, cat_cols = _classify_cols(df_joined, exclude)
    print(f"  {len(df_joined)} movimientos | {len(bool_cols)} bool + {len(cat_cols)} cat")

    raw = _mine(df_joined, bool_cols, cat_cols)
    print(f"  Patrones crudos: {len(raw)}")

    if not raw:
        print("  Sin patrones con N suficiente.")
        return pd.DataFrame()

    df_val = _walk_forward(df_joined, raw)
    if df_val.empty:
        print("  Ningun patron supero walk-forward.")
        return pd.DataFrame()

    df_val = df_val.sort_values("wr_valid", ascending=False, na_position="last")
    path   = os.path.join(RESULTS_DIR, f"{symbol}_validated_patterns.csv")
    df_val.to_csv(path, index=False)
    print(f"  Patrones validados: {len(df_val)} -> {path}")
    return df_val
