import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import sys

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# config
MIN_WR_UNI    = 0.57
MIN_OCC       = 3      # minimo trades para incluir combo
HOLD_BARS_1M  = 240    # 4 horas
TP_PCT        = 0.015
SL_PCT        = 0.007
MAX_COMBO_LEN = 4

symbol     = "NVDA"
output_dir = Path("output") / symbol

print("Cargando datos...")
sm  = pd.read_csv(output_dir / "state_matrix.csv", index_col=0)
uni = pd.read_csv(output_dir / "univariate.csv")
uni.columns = uni.columns.str.strip().str.lower()
sm.columns  = sm.columns.str.strip()
print(f"  state_matrix : {sm.shape}")

# señales base
wr_col  = "wr_long_120m"
n_col   = "ocurrencias"
var_col = "variable"
mask    = (uni[wr_col] >= MIN_WR_UNI) & (uni[n_col] >= 500)
signals = uni.loc[mask, var_col].tolist()
signals = [s for s in signals if s in sm.columns]
print(f"  Señales validas: {len(signals)} → {signals}")

# precio
close_col = next((c for c in sm.columns if c.lower() == "close"), None)
if not close_col:
    close_col = [c for c in sm.columns if "close" in c.lower()][0]
prices = sm[close_col].values
n_bars = len(prices)

def simulate_combo(cols, direction, prices, sm):
    mask = sm[cols[0]].astype(bool)
    for c in cols[1:]:
        mask = mask & sm[c].astype(bool)

    entry_idx = np.where(mask.values[:-HOLD_BARS_1M])[0]
    if len(entry_idx) < MIN_OCC:
        return None

    pnls = []
    exits = []
    last_exit = -1

    for i in entry_idx:
        if i <= last_exit:
            continue
        ep = prices[i]
        if ep <= 0 or np.isnan(ep):
            continue

        end = min(i + HOLD_BARS_1M + 1, n_bars)
        fut = prices[i+1:end]

        if direction == "LONG":
            tp_price = ep * (1 + TP_PCT)
            sl_price = ep * (1 - SL_PCT)
            tp_hits  = np.where(fut >= tp_price)[0]
            sl_hits  = np.where(fut <= sl_price)[0]
        else:
            tp_price = ep * (1 - TP_PCT)
            sl_price = ep * (1 + SL_PCT)
            tp_hits  = np.where(fut <= tp_price)[0]
            sl_hits  = np.where(fut >= sl_price)[0]

        first_tp = tp_hits[0] if len(tp_hits) else HOLD_BARS_1M
        first_sl = sl_hits[0] if len(sl_hits) else HOLD_BARS_1M

        if first_tp < first_sl:
            xp = tp_price; xr = "TP"; xb = i + first_tp + 1
        elif first_sl < first_tp:
            xp = sl_price; xr = "SL"; xb = i + first_sl + 1
        else:
            xp = prices[min(i + HOLD_BARS_1M, n_bars-1)]; xr = "TIME"; xb = i + HOLD_BARS_1M

        pnl = (xp - ep)/ep if direction == "LONG" else (ep - xp)/ep
        pnls.append(pnl)
        exits.append(xr)
        last_exit = xb

    n = len(pnls)
    if n < MIN_OCC:
        return None

    pnls  = np.array(pnls)
    wr    = (pnls > 0).mean()
    avg   = pnls.mean()
    total = pnls.sum()
    edge  = wr * TP_PCT - (1-wr) * SL_PCT
    tp_r  = exits.count("TP") / n

    return {
        "n_trades": n,
        "win_rate": round(wr, 4),
        "avg_pnl":  round(avg, 6),
        "total_pnl":round(total, 4),
        "tp_rate":  round(tp_r, 3),
        "edge":     round(edge, 6),
    }

# analizar combos
print()
print("Analizando combinaciones...")
all_results = []

for combo_len in range(1, MAX_COMBO_LEN + 1):
    combos = list(combinations(signals, combo_len))
    found  = 0
    for combo in combos:
        r = simulate_combo(list(combo), "LONG", prices, sm)
        if r:
            all_results.append({
                "signals":   " + ".join(combo),
                "n_signals": combo_len,
                "direction": "LONG",
                **r
            })
            found += 1
    print(f"  Longitud {combo_len}: {len(combos):,} combos → {found} validas")

if not all_results:
    print()
    print("  Sin resultados con MIN_OCC=3. Mostrando estadisticas brutas...")
    for s in signals:
        mask = sm[s].astype(bool)
        n_entries = mask.sum()
        print(f"    {s}: {n_entries} entradas activas")
else:
    df = pd.DataFrame(all_results).sort_values("edge", ascending=False).reset_index(drop=True)
    out = output_dir / "combo_analysis.csv"
    df.to_csv(out, index=False)
    print(f"  Guardado: {out} ({len(df)} combos)")

    print()
    print("=" * 85)
    print("TOP 25 COMBINACIONES POR EDGE")
    print("=" * 85)
    print(f"{'#':<3} {'N':<4} {'SEÑALES':<55} {'TRADES':<7} {'WR':<7} {'EDGE':<9} PnL_TOT")
    print("-" * 85)
    for i, row in df.head(25).iterrows():
        sigs = row["signals"][:53]
        print(f"{i+1:<3} {row['n_signals']:<4} {sigs:<55} {row['n_trades']:<7} {row['win_rate']:.1%}  {row['edge']:+.5f}  {row['total_pnl']:+.3f}")

    print()
    print("TOP 10 ALTA FRECUENCIA (edge > 0, mas trades)")
    print("-" * 85)
    df_freq = df[df["edge"] > 0].sort_values("n_trades", ascending=False)
    for i, row in df_freq.head(10).iterrows():
        sigs = row["signals"][:53]
        print(f"  N={row['n_signals']} {sigs:<55} T={row['n_trades']:<5} WR={row['win_rate']:.1%} Edge={row['edge']:+.5f}")
