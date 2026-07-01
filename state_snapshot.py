"""
state_snapshot.py
Para cada movimiento detectado, captura el estado exacto de todos los
indicadores en el momento de entrada (inicio) y en el momento de agotamiento (fin).
Usa las columnas REALES del state_matrix.csv.
"""
import pandas as pd
import numpy as np

TFS        = ["1M","5M","15M","30M","1H","4H","1D","1W"]
INDICATORS = {
    "macd": [
        "macd_hist", "macd_hist_delta", "macd_above_zero",
        "macd_above_sig", "macd_e1_bull", "macd_e1_bear",
        "macd_e2_bull", "macd_e2_bear",
        "macd_hist_bars_rising", "macd_hist_bars_falling",
    ],
    "stoch": [
        "stoch_K", "stoch_D", "stoch_KD_above",
        "stoch_ob", "stoch_os", "stoch_bars_extreme",
        "stoch_A_bull", "stoch_A_bear",
        "stoch_B_bull", "stoch_B_bear",
    ],
    "ema8":   ["dist_ema8_pct",  "above_ema8",  "near_ema8",  "cross_bull_ema8",  "cross_bear_ema8",  "reject_bull_ema8",  "reject_bear_ema8"],
    "ema21":  ["dist_ema21_pct", "above_ema21", "near_ema21", "cross_bull_ema21", "cross_bear_ema21", "reject_bull_ema21", "reject_bear_ema21"],
    "sma50":  ["dist_sma50_pct", "above_sma50", "near_sma50", "cross_bull_sma50", "cross_bear_sma50", "reject_bull_sma50", "reject_bear_sma50"],
    "sma200": ["dist_sma200_pct","above_sma200","near_sma200","cross_bull_sma200","cross_bear_sma200","reject_bull_sma200","reject_bear_sma200"],
}

def _build_col_list(sm_cols: list) -> list:
    """Construye lista de columnas de estado que existen realmente en el CSV."""
    wanted = []
    for tf in TFS:
        for ind, fields in INDICATORS.items():
            for f in fields:
                col = f"{tf}_{f}"
                if col in sm_cols:
                    wanted.append(col)
    return wanted

def snapshot_states(sm: pd.DataFrame, movements: pd.DataFrame) -> tuple:
    """
    Devuelve (entry_states, exit_states).
    Cada fila = un movimiento. Columnas = estado de cada indicador en ese instante.
    """
    state_cols = _build_col_list(list(sm.columns))
    print(f"  Columnas de estado encontradas: {len(state_cols)}")

    entry_rows = []
    exit_rows  = []

    for _, mv in movements.iterrows():
        bs = int(mv["bar_start"])
        be = int(mv["bar_end"])

        # ventana de contexto: 3 barras antes del inicio para estados "previos"
        ctx_start = max(0, bs - 3)

        base = {
            "mov_id":     mv["id"],
            "direction":  mv["direction"],
            "pct_move":   mv["pct_move"],
            "duration_bars": mv["duration_bars"],
            "duration_hrs":  mv["duration_hrs"],
            "price_start":   mv["price_start"],
            "price_end":     mv["price_end"],
            "ts_start":      mv["ts_start"],
            "ts_end":        mv["ts_end"],
        }

        # snapshot en el inicio del movimiento
        entry_row = dict(base)
        for col in state_cols:
            try:
                entry_row[col] = sm[col].iloc[bs]
            except Exception:
                entry_row[col] = np.nan
        entry_rows.append(entry_row)

        # snapshot en el fin del movimiento (agotamiento)
        exit_row = dict(base)
        for col in state_cols:
            try:
                exit_row[col] = sm[col].iloc[be]
            except Exception:
                exit_row[col] = np.nan
        exit_rows.append(exit_row)

    entry_df = pd.DataFrame(entry_rows)
    exit_df  = pd.DataFrame(exit_rows)
    print(f"  entry_states: {entry_df.shape} | exit_states: {exit_df.shape}")
    return entry_df, exit_df
