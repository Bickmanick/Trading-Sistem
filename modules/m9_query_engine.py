"""
MODULO 9 - QUERY ENGINE
Compara el estado actual del mercado con los patrones historicos validados.
Usa el state matrix actual (ultima fila) con los mismos nombres de columna
que usa m8, garantizando que la comparacion sea correcta.

Output: consola + data/results/{symbol}_current_state.json
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from config import DATA_DIR

RESULTS_DIR = os.path.join(DATA_DIR, "results")


def _current_state(state_matrix: pd.DataFrame) -> dict:
    """
    Lee la ultima fila del state matrix completo (ya tiene todos los TFs
    prefijados correctamente por state_engine.build_state_matrix).
    """
    if state_matrix.empty:
        return {"timestamp": datetime.now(timezone.utc).isoformat()}

    row = state_matrix.sort_index().iloc[-1]
    state = {"timestamp": str(row.name)}
    for col in state_matrix.columns:
        v = row[col]
        if isinstance(v, float) and np.isnan(v):
            state[col] = None
        else:
            state[col] = v
    return state


def run_query(symbol: str, state_matrix: pd.DataFrame,
             validated: pd.DataFrame):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cur = _current_state(state_matrix)
    ts  = cur.get("timestamp", "")[:19]

    print(f"\n{'='*60}")
    print(f"  ESTADO ACTUAL --- {symbol}  |  {ts}")
    print(f"{'='*60}")

    if validated.empty:
        print("  Sin patrones validados todavia.")
    else:
        matches = []
        for _, p in validated.iterrows():
            col  = p.get("variables")
            raw  = str(p.get("patron", ""))
            if "==" not in raw or col not in cur:
                continue
            val      = raw.split("==")[1]
            tipo     = p.get("tipo", "cat")
            cur_val  = cur.get(col)

            if cur_val is None:
                continue

            if tipo == "bool":
                match = (float(cur_val) == float(val))
            else:
                match = (str(cur_val) == str(val))

            if match:
                matches.append(p)

        if not matches:
            print("\n  Sin coincidencias con el estado actual.")
        else:
            print(f"\n  {len(matches)} PATRON(ES) ACTIVO(S):\n")
            for p in matches:
                wr  = p.get("wr_valid", np.nan)
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
        json.dump({k: (None if isinstance(v, float) and np.isnan(v) else
                       (v.item() if hasattr(v, 'item') else v))
                   for k, v in cur.items()}, f, indent=2, default=str)
    print(f"  Estado guardado: {path}")
