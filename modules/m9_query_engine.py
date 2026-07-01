"""
MODULO 9 - QUERY ENGINE
Estado actual del mercado vs patrones historicos validados.
Output: consola + data/results/{symbol}_current_state.json
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from config import DATA_DIR

RESULTS_DIR = os.path.join(DATA_DIR, "results")
SKIP_COLS   = {"open", "high", "low", "close", "volume", "swing_type", "swing_price"}


def _current_state(tfs_ind):
    state = {"timestamp": datetime.now(timezone.utc).isoformat()}
    for tf_name, df in tfs_ind.items():
        if df.empty:
            continue
        row = df.sort_index().iloc[-1]
        for col in df.columns:
            if col not in SKIP_COLS:
                v = row.get(col, np.nan)
                state[f"{tf_name}_{col}"] = None if (isinstance(v, float) and np.isnan(v)) else v
    return state


def run_query(symbol, tfs_ind, validated):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cur = _current_state(tfs_ind)
    print(f"\n{'='*60}")
    print(f"  ESTADO ACTUAL --- {symbol}  |  {cur['timestamp'][:19]}")
    print(f"{'='*60}")
    if validated.empty:
        print("  Sin patrones validados todavia (necesitas mas historico).")
    else:
        matches = []
        for _, p in validated.iterrows():
            col = p.get("variables")
            val = str(p.get("patron", "")).split("==")[-1] if "==" in str(p.get("patron", "")) else None
            if col and val and str(cur.get(f"{col}", "")) == val:
                matches.append(p)
        if not matches:
            print("\n  Sin coincidencias con el estado actual.")
        else:
            print(f"\n  {len(matches)} PATRON(ES) ACTIVO(S):\n")
            for p in matches:
                wr = p.get("wr_valid", np.nan)
                wr_str = f"{wr:.1%}" if isinstance(wr, float) and not np.isnan(wr) else "pendiente"
                print(f"  [{p['direccion']}] {p['patron']}")
                print(f"    WR validado : {wr_str}")
                print(f"    N historico : {p.get('N', '?')}")
                print(f"    Mag media   : {p.get('mag_media', '?')}%")
                print(f"    Fib rebote  : {p.get('rebote_fib_frecuente', '?')}")
                print(f"    Estado      : {p.get('estado', '?')}")
                print()
    path = os.path.join(RESULTS_DIR, f"{symbol}_current_state.json")
    with open(path, "w") as f:
        json.dump(cur, f, indent=2, default=str)
    print(f"  Estado guardado: {path}")
