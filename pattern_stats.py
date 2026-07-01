"""
pattern_stats.py v2
Patrones basados en columnas de 1H y 4H (varían en escala de horas).
Para datasets de 2-3 meses: usar TFs 1H/4H en vez de 1D/1W.
"""
import pandas as pd
import numpy as np

# Columnas que varían en escala de horas (no días) — relevantes para 3 meses de datos
PATTERN_COLS_ENTRY = [
    # MACD 4H — estructura del movimiento
    "4H_macd_above_zero",
    "4H_macd_above_sig",
    "4H_macd_e2_bull",
    "4H_macd_e2_bear",
    "4H_macd_hist_bars_rising",
    "4H_macd_hist_bars_falling",
    # MACD 1H — confirmación
    "1H_macd_above_zero",
    "1H_macd_above_sig",
    "1H_macd_e2_bull",
    "1H_macd_e2_bear",
    # Stoch 1H — zona y giro
    "1H_stoch_ob",
    "1H_stoch_os",
    "1H_stoch_A_bull",
    "1H_stoch_A_bear",
    # Stoch 5M — trigger
    "5M_stoch_A_bull",
    "5M_stoch_A_bear",
    "5M_stoch_ob",
    "5M_stoch_os",
    # Medias 4H
    "4H_above_ema21",
    "4H_above_sma50",
    # Medias 1H
    "1H_above_ema8",
    "1H_above_ema21",
]

PATTERN_COLS_EXIT = [
    "4H_stoch_ob",
    "4H_stoch_os",
    "4H_stoch_A_bear",
    "4H_stoch_A_bull",
    "4H_macd_hist_bars_falling",
    "4H_macd_hist_bars_rising",
    "1H_stoch_ob",
    "1H_stoch_os",
    "1H_stoch_A_bear",
    "1H_stoch_A_bull",
    "1H_macd_hist_bars_falling",
    "1H_macd_hist_bars_rising",
    "5M_stoch_A_bear",
    "5M_stoch_A_bull",
]

def _build_pattern_key(row: pd.Series, cols: list) -> str:
    parts = []
    for col in cols:
        if col not in row.index:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        if isinstance(val, (bool, np.bool_)):
            parts.append(f"{col}={int(val)}")
        elif isinstance(val, (int, np.integer)):
            if int(val) != 0:  # ignorar ceros para compactar la clave
                parts.append(f"{col}={int(val)}")
        elif isinstance(val, float):
            if "dist_" in col:
                rng = ("FAR_BELOW" if val < -3 else
                       "NEAR_BELOW" if val < -0.5 else
                       "AT" if val < 0.5 else
                       "NEAR_ABOVE" if val < 3 else "FAR_ABOVE")
                parts.append(f"{col}={rng}")
    return "|".join(parts) if parts else "NO_SIGNAL"

def compute_pattern_stats(entry_df: pd.DataFrame, label: str = "entry") -> pd.DataFrame:
    cols_to_use = PATTERN_COLS_ENTRY if label == "entry" else PATTERN_COLS_EXIT
    cols_to_use = [c for c in cols_to_use if c in entry_df.columns]

    if not cols_to_use:
        print(f"  AVISO: ninguna columna encontrada para {label}")
        return pd.DataFrame()

    print(f"  Columnas de patrón ({label}): {len(cols_to_use)}")

    df = entry_df.copy()
    df["pattern_key"] = df.apply(lambda r: _build_pattern_key(r, cols_to_use), axis=1)

    results = []
    for key, group in df.groupby("pattern_key"):
        n = len(group)
        if n < 3:
            continue

        pcts      = group["pct_move"].values
        durs      = group["duration_hrs"].values
        mfes      = group["mfe_pct"].values if "mfe_pct" in group.columns else np.abs(pcts)
        long_c    = (group["direction"] == "LONG").sum()
        short_c   = (group["direction"] == "SHORT").sum()
        dominant  = "LONG" if long_c >= short_c else "SHORT"
        pct_align = (group["direction"] == dominant).mean()

        results.append({
            "pattern":       key[:150],
            "n":             n,
            "dominant_dir":  dominant,
            "pct_aligned":   round(pct_align * 100, 1),
            "pct_move_med":  round(np.median(np.abs(pcts)), 3),
            "pct_move_mean": round(np.mean(np.abs(pcts)), 3),
            "mfe_med":       round(np.median(mfes), 3),
            "dur_hrs_med":   round(np.median(durs), 1),
        })

    df_out = pd.DataFrame(results)
    if len(df_out):
        df_out = df_out.sort_values(["pct_aligned","n"], ascending=False).reset_index(drop=True)
        print(f"  Patrones únicos con ≥3 ocurrencias: {len(df_out)}")
    return df_out
