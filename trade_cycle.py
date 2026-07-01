import pandas as pd
import numpy as np
from pathlib import Path
import sys

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MIN_WR        = 0.57
MIN_OCC       = 200
HOLD_BARS_1M  = 240    # 4 horas
TP_PCT        = 0.015  # 1.5%
SL_PCT        = 0.007  # 0.7%

def run_trade_cycle(state_matrix, patterns, univariate, symbol, output_dir):

    uni = univariate.copy()
    uni.columns = uni.columns.str.strip().str.lower()
    available = list(uni.columns)

    col_map = {}
    for c in available:
        if c == "variable":                           col_map["variable"] = c
        elif c in ("ocurrencias", "n", "count"):      col_map["n"] = c
        elif "120m" in c and "long" in c:             col_map["wr_4h_long"] = c
        elif "120m" in c and "short" in c:            col_map["wr_4h_short"] = c
        elif "60m" in c and "long" in c and "wr_4h_long" not in col_map:
                                                      col_map["wr_4h_long"] = c
        elif "60m" in c and "short" in c and "wr_4h_short" not in col_map:
                                                      col_map["wr_4h_short"] = c

    var_col  = col_map.get("variable", "variable")
    wr_l_col = col_map.get("wr_4h_long")
    wr_s_col = col_map.get("wr_4h_short")
    n_col    = col_map.get("n")

    print(f"  WR long col: {wr_l_col} | WR short col: {wr_s_col}")

    def get_signals(wr_col, min_wr):
        if wr_col is None: return []
        mask = uni[wr_col] >= min_wr
        if n_col: mask &= uni[n_col] >= MIN_OCC
        return uni.loc[mask, var_col].tolist()

    long_signals  = get_signals(wr_l_col, MIN_WR)
    short_signals = get_signals(wr_s_col, MIN_WR)
    if not long_signals and not short_signals:
        long_signals  = get_signals(wr_l_col, 0.55)
        short_signals = get_signals(wr_s_col, 0.55)

    print(f"  Senales LONG : {len(long_signals)}")
    print(f"  Senales SHORT: {len(short_signals)}")

    sm = state_matrix.copy()
    sm.columns = sm.columns.str.strip()
    close_col = next((c for c in sm.columns if c.lower() == "close"), None)
    if close_col is None:
        close_col = [c for c in sm.columns if "close" in c.lower()][0]

    sm_lower = {c.lower(): c for c in sm.columns}
    def find_col(name):
        if name in sm.columns: return name
        return sm_lower.get(name.lower())

    long_cols  = [find_col(s) for s in long_signals  if find_col(s)]
    short_cols = [find_col(s) for s in short_signals if find_col(s)]

    long_mask  = sm[long_cols].any(axis=1)  if long_cols  else pd.Series(False, index=sm.index)
    short_mask = sm[short_cols].any(axis=1) if short_cols else pd.Series(False, index=sm.index)

    print(f"  Velas LONG : {long_mask.sum():,} | SHORT: {short_mask.sum():,}")

    prices   = sm[close_col].values
    n_bars   = len(prices)
    trades   = []
    in_trade = False
    trade_end = 0

    for i in range(n_bars - HOLD_BARS_1M):
        if in_trade and i < trade_end:
            continue
        in_trade = False

        ep = prices[i]
        if ep <= 0 or np.isnan(ep): continue

        direction = None
        if long_mask.iloc[i]:   direction = "LONG"
        elif short_mask.iloc[i]: direction = "SHORT"
        if direction is None: continue

        tp = ep*(1+TP_PCT) if direction=="LONG" else ep*(1-TP_PCT)
        sl = ep*(1-SL_PCT) if direction=="LONG" else ep*(1+SL_PCT)
        xp = prices[min(i+HOLD_BARS_1M, n_bars-1)]; xr="TIME"; xb=i+HOLD_BARS_1M

        for j in range(i+1, min(i+HOLD_BARS_1M+1, n_bars)):
            p = prices[j]
            if direction=="LONG":
                if p>=tp: xp,xr,xb=tp,"TP",j; break
                if p<=sl: xp,xr,xb=sl,"SL",j; break
            else:
                if p<=tp: xp,xr,xb=tp,"TP",j; break
                if p>=sl: xp,xr,xb=sl,"SL",j; break

        pnl = (xp-ep)/ep if direction=="LONG" else (ep-xp)/ep
        trades.append({"bar_entry":i,"bar_exit":xb,"direction":direction,
                       "entry_price":round(ep,4),"exit_price":round(xp,4),
                       "pnl_pct":round(pnl,5),"exit_reason":xr})
        in_trade=True; trade_end=xb

    if not trades:
        print("  No se generaron trades.")
        return

    df = pd.DataFrame(trades)
    wr = (df["pnl_pct"]>0).mean()
    print()
    print(f"  ====== RESUMEN {symbol} ======")
    print(f"  Total trades   : {len(df):,}")
    print(f"  Win Rate       : {wr:.1%}")
    print(f"  PnL total      : {df['pnl_pct'].sum():+.2%}")
    print(f"  PnL medio      : {df['pnl_pct'].mean():+.4%}")
    print(f"  Salidas TP/SL/T: {(df.exit_reason=='TP').sum()} / {(df.exit_reason=='SL').sum()} / {(df.exit_reason=='TIME').sum()}")
    print(f"  ============================")

    df.to_csv(output_dir/"trades.csv", index=False)
    print(f"  Guardado: trades.csv ({len(df):,} trades)")
    return df
