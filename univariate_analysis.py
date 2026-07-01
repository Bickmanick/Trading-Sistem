import numpy as np
import pandas as pd
from config import FORWARD_WINDOWS, EDGE_WINRATE, EDGE_MIN_OCC

EVENT_SUFFIXES = (
    "_macd_e1_bull","_macd_e1_bear","_macd_e2_bull","_macd_e2_bear",
    "_stoch_A_bull","_stoch_A_bear","_stoch_B_bull","_stoch_B_bear",
    "_cross_bull_ema21","_cross_bear_ema21","_cross_bull_sma200","_cross_bear_sma200",
    "_reject_bull_ema21","_reject_bear_ema21","_reject_bull_sma200","_reject_bear_sma200",
    "_extreme_bull_sma200","_extreme_bear_sma200",
    "_fib0382_in_zone","_fib05_in_zone","_fib0618_in_zone",
    "_fib0382_reject_bull","_fib05_reject_bull","_fib0618_reject_bull",
    "_fib0382_reject_bear","_fib05_reject_bear","_fib0618_reject_bear",
    "_fib05_conf_sma200","_fib0382_conf_ema21","_stoch_ob","_stoch_os",
)

def univariate_analysis(state):
    close = state["close"] if "close" in state.columns else             state[[c for c in state.columns if c=="close" or c.endswith("_close")][0]]
    ecols = [c for c in state.columns
             if any(c.endswith(s) for s in EVENT_SUFFIXES)
             and state[c].dtype in ["int64","float64","int32"]]
    print(f"  Analizando {len(ecols)} variables...")
    records = []
    for col in ecols:
        mask = state[col].fillna(0).astype(bool)
        occ  = mask.sum()
        if occ < EDGE_MIN_OCC:
            continue
        row = {"variable":col, "ocurrencias":occ}
        best_wr = 0
        for w in FORWARD_WINDOWS:
            ret = close.shift(-w)/close - 1
            r   = ret[mask]
            wl  = float((r>0).mean())
            ws  = float((r<0).mean())
            row[f"wr_long_{w}m"]  = round(wl, 4)
            row[f"wr_short_{w}m"] = round(ws, 4)
            row[f"pnl_mean_{w}m"] = round(float(r.mean()*100), 4)
            best_wr = max(best_wr, wl, ws)
        row["best_wr"]  = round(best_wr, 4)
        row["has_edge"] = best_wr >= EDGE_WINRATE and occ >= EDGE_MIN_OCC
        records.append(row)
    df = pd.DataFrame(records).sort_values("best_wr", ascending=False)
    evars = df[df["has_edge"]]["variable"].tolist()
    print(f"  {len(evars)} variables con ventaja >= {EDGE_WINRATE*100:.0f}%")
    return evars, df
