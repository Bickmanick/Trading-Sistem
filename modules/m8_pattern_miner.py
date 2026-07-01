"""
MODULO 8 - PATTERN MINER con validacion walk-forward

Mina columnas:
  - booleanas (True/False, 0/1)  -> las mas utiles del state matrix
  - categoricas string           -> estructura, bos_event, choch_event

Filtro walk-forward:
  - Train 75% / Validacion 25% por tiempo
  - N >= 30 en train
  - WR validacion >= 55%
  - Overfitting (train - valid) <= 10%

Output: data/results/{symbol}_validated_patterns.csv
"""
import os
import pandas as pd
import numpy as np
from config import DATA_DIR

RESULTS_DIR  = os.path.join(DATA_DIR, "results")
TRAIN_RATIO  = 0.75
MIN_N        = 30
MIN_WR_VALID = 0.55
MAX_OVERFIT  = 0.10


def _classify_cols(df: pd.DataFrame, exclude: set) -> tuple:
    """Separa columnas en booleanas y categoricas string."""
    bool_cols = []
    cat_cols  = []
    for c in df.columns:
        if c in exclude:
            continue
        dtype = str(df[c].dtype)
        if dtype in ("bool", "boolean"):
            bool_cols.append(c)
            continue
        if dtype in ("int64", "int32", "uint8"):
            uniq = df[c].dropna().unique()
            if set(uniq).issubset({0, 1, True, False}):
                bool_cols.append(c)
            continue
        if dtype == "float64":
            uniq = df[c].dropna().unique()
            if set(uniq).issubset({0.0, 1.0}):
                bool_cols.append(c)
            continue
        if dtype == "object":
            cat_cols.append(c)
    return bool_cols, cat_cols


def _mine(df: pd.DataFrame, bool_cols: list, cat_cols: list) -> list:
    records = []

    # --- Booleanas: cuando la variable es True (1) ---
    for col in bool_cols:
        subset = df[df[col].astype(float) == 1.0]
        if len(subset) < MIN_N:
            continue
        for direction in ["LONG", "SHORT"]:
            sub_d = subset[subset["direccion"] == direction]
            if len(sub_d) < MIN_N:
                continue
            wr = (sub_d["continuacion"] == "CONTINUACION").mean()
            records.append({
                "patron":    f"{col}==1",
                "variables": col,
                "tipo":      "bool",
                "direccion": direction,
                "N":         len(sub_d),
                "winrate":   round(wr, 4),
                "mag_media": round(sub_d["magnitud_pct"].mean(), 4),
                "mag_std":   round(sub_d["magnitud_pct"].std(),  4),
                "dur_media": round(sub_d["duracion_min"].mean(),  1),
                "rebote_fib_frecuente": sub_d["rebote_fib"].mode().iloc[0]
                    if not sub_d["rebote_fib"].mode().empty else np.nan,
            })

    # --- Categoricas: cada valor unico ---
    for col in cat_cols:
        for val in df[col].dropna().unique():
            if str(val) == "NONE":
                continue  # NONE es el estado neutro, no aporta info
            subset = df[df[col] == val]
            for direction in ["LONG", "SHORT"]:
                sub_d = subset[subset["direccion"] == direction]
                if len(sub_d) < MIN_N:
                    continue
                wr = (sub_d["continuacion"] == "CONTINUACION").mean()
                records.append({
                    "patron":    f"{col}=={val}",
                    "variables": col,
                    "tipo":      "cat",
                    "direccion": direction,
                    "N":         len(sub_d),
                    "winrate":   round(wr, 4),
                    "mag_media": round(sub_d["magnitud_pct"].mean(), 4),
                    "mag_std":   round(sub_d["magnitud_pct"].std(),  4),
                    "dur_media": round(sub_d["duracion_min"].mean(),  1),
                    "rebote_fib_frecuente": sub_d["rebote_fib"].mode().iloc[0]
                        if not sub_d["rebote_fib"].mode().empty else np.nan,
                })
    return records


def _walk_forward(df: pd.DataFrame, patterns: list) -> pd.DataFrame:
    cutoff = df["timestamp_inicio"].quantile(TRAIN_RATIO)
    df_t   = df[df["timestamp_inicio"] <= cutoff]
    df_v   = df[df["timestamp_inicio"] >  cutoff]
    validated = []

    for p in patterns:
        col  = p["variables"]
        val  = p["patron"].split("==")[1]
        dir_ = p["direccion"]
        tipo = p.get("tipo", "cat")

        if tipo == "bool":
            sub_t = df_t[(df_t[col].astype(float) == float(val)) & (df_t["direccion"] == dir_)]
            sub_v = df_v[(df_v[col].astype(float) == float(val)) & (df_v["direccion"] == dir_)]
        else:
            sub_t = df_t[(df_t[col].astype(str) == str(val)) & (df_t["direccion"] == dir_)]
            sub_v = df_v[(df_v[col].astype(str) == str(val)) & (df_v["direccion"] == dir_)]

        if len(sub_t) < MIN_N:
            continue
        wr_t = (sub_t["continuacion"] == "CONTINUACION").mean()

        if len(sub_v) < 10:
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

    df_out    = df_outcomes[["mov_id", "continuacion", "rebote_fib",
                              "mfe_pct", "mae_pct"]].copy()
    df_joined = df_start.merge(df_out, on="mov_id", how="inner")
    df_joined["timestamp_inicio"] = pd.to_datetime(
        df_joined["timestamp_inicio"], utc=True)

    exclude = {"mov_id", "tf", "direccion", "magnitud_pct", "duracion_min",
               "tf_parent_id", "continuacion", "rebote_fib", "mfe_pct",
               "mae_pct", "timestamp_inicio", "ts"}

    bool_cols, cat_cols = _classify_cols(df_joined, exclude)
    print(f"  {len(df_joined)} movimientos | {len(bool_cols)} bool + {len(cat_cols)} cat = {len(bool_cols)+len(cat_cols)} variables")

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
