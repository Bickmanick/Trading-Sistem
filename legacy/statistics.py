import os
import pandas as pd

def compute_statistics(trades: pd.DataFrame, patterns: pd.DataFrame) -> dict:
    if trades.empty:
        return {}
    wr      = (trades["pnl_pct"] > 0).mean()
    avg_pnl = trades["pnl_pct"].mean()
    return {"n_trades": len(trades), "winrate": round(wr, 4), "avg_pnl": round(avg_pnl, 4)}

def save_statistics(stats: dict, output_dir: str, symbol: str):
    path = os.path.join(output_dir, symbol, "statistics.csv")
    pd.DataFrame([stats]).to_csv(path, index=False)
    print(f"  Estadisticas guardadas: {path}")
