import pandas as pd
import os

def compute_statistics(trades, patterns):
    if trades.empty:
        return {}
    stats = {}
    for grp in ["entry_regime","thesis_quality","direction","exit_type"]:
        if grp in trades.columns:
            stats[f"by_{grp}"] = trades.groupby(grp).agg(
                n_trades=("pnl_pct","count"),
                wr=("pnl_pct", lambda x:(x>0).mean()),
                avg_pnl=("pnl_pct","mean"),
                avg_mae=("mae_pct","mean") if "mae_pct" in trades.columns else ("pnl_pct","count"),
                avg_mfe=("mfe_pct","mean") if "mfe_pct" in trades.columns else ("pnl_pct","count"),
            ).round(4)
    if not patterns.empty:
        rank = ["n_neighbors","regime","thesis_long","thesis_short","fib_conf"]
        for w in [30,60,120]:
            for d in ["long","short"]:
                c = f"wr_{d}_{w}m"
                if c in patterns.columns: rank.append(c)
        rank = [c for c in rank if c in patterns.columns]
        df_r = patterns[rank].copy()
        if "wr_short_60m" in df_r.columns and "wr_long_60m" in df_r.columns:
            df_r["best_wr_60m"] = df_r[["wr_long_60m","wr_short_60m"]].max(axis=1)
            df_r["score"]       = df_r["n_neighbors"] * df_r["best_wr_60m"]
            df_r = df_r.sort_values("score", ascending=False)
        stats["patterns_ranked"] = df_r
    return stats

def save_statistics(stats, output_dir, symbol):
    out = os.path.join(output_dir, symbol)
    os.makedirs(out, exist_ok=True)
    for name, df in stats.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            p = os.path.join(out, f"{name}.csv")
            df.to_csv(p)
            print(f"  Guardado: {p}")
