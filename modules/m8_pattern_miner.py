"""
MODULO 8 - PATTERN MINER con validacion walk-forward

Mejoras respecto a version anterior:
  - Filtra movimientos ULTIMO antes de minar (no tienen siguiente real)
  - Umbral N adaptativo: N>=5 para TFs altos (4H,1D,1W), N>=30 para bajos
  - Mina tambien por TF como variable explicita
  - Incluye alineamiento_3tf y cierre_tipo como variables
  - Calcula magnitud esperada por TF y direccion
  - Incluye fib_agotamiento en el resumen del patron

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

# Umbral minimo de ocurrencias por TF
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
    return {
        "N":           len(sub),
        "winrate":     round((sub["continuacion"] == "CONTINUACION").mean(), 4),
        "mag_media":   round(sub["magnitud_pct"].mean(), 4),
        "mag_std":     round(sub["magnitud_pct"].std(),  4),
        "dur_media":   round(sub["duracion_min"].mean(),  1),
        "mfe_media":   round(sub["mfe_pct"].mean(), 4) if "mfe_pct" in sub else np.nan,
        "mae_media":   round(sub["mae_pct"].mean(), 4) if "mae_pct" in sub else np.nan,
        "rebote_fib_frecuente":    sub["rebote_fib"].mode().iloc[0]
            if "rebote_fib" in sub and not sub["rebote_fib"].mode().empty else np.nan,
        "fib_agot_frecuente":      sub["fib_agotamiento"].mode().iloc[0]
            if "fib_agotamiento" in sub and not sub["fib_agotamiento"].mode().empty else np.nan,
    }


def _mine(df: pd.DataFrame, bool_cols: list, cat_cols: list) -> list:
    records = []
    tfs = df["tf"].unique() if "tf" in df.columns else [None]

    def _add(col, val, tipo, direction, subset, tf=None):
        mn = _min_n(tf) if tf else 10
        sub_d = subset[subset["direccion"] == direction]
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

    # --- Global (todos los TFs) ---
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

    # --- Por TF ---
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

    # Unir estado al inicio del movimiento con sus outcomes
    keep_out = ["mov_id", "continuacion", "rebote_fib", "fib_agotamiento",
                "mfe_pct", "mae_pct", "cierre_tipo", "alineamiento_3tf"]
    keep_out = [c for c in keep_out if c in df_outcomes.columns]
    df_out    = df_outcomes[keep_out].copy()

    # Filtrar ULTIMO — no tienen siguiente movimiento real
    df_out = df_out[df_out["continuacion"] != "ULTIMO"]

    df_joined = df_start.merge(df_out, on="mov_id", how="inner")
    df_joined["timestamp_inicio"] = pd.to_datetime(
        df_joined["timestamp_inicio"], utc=True)

    exclude = {"mov_id", "tf", "direccion", "magnitud_pct", "duracion_min",
               "tf_parent_id", "continuacion", "rebote_fib", "fib_agotamiento",
               "mfe_pct", "mae_pct", "timestamp_inicio", "ts",
               "cierre_tipo", "alineamiento_3tf"}

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
