"""
MODULO 9 - QUERY ENGINE

Mejoras:
  - Busca coincidencias globales Y por TF especifico
  - Muestra el TF del patron junto al resultado
  - Agrupa patrones por direccion y TF para lectura clara
  - Incluye fib_agot_frecuente y mfe/mae en el resumen
  - Indica nivel de alineamiento cuando el patron lo requiere

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
    if state_matrix.empty:
        return {"timestamp": datetime.now(timezone.utc).isoformat()}
    row   = state_matrix.sort_index().iloc[-1]
    state = {"timestamp": str(row.name)}
    for col in state_matrix.columns:
        v = row[col]
        state[col] = None if (isinstance(v, float) and np.isnan(v)) else v
    return state


def _match_pattern(p: pd.Series, cur: dict) -> bool:
    col = p.get("variables")
    raw = str(p.get("patron", ""))
    if "==" not in raw or col not in cur:
        return False
    val     = raw.split("==")[1].split("@")[0]
    tipo    = p.get("tipo", "cat")
    cur_val = cur.get(col)
    if cur_val is None:
        return False
    try:
        return (float(cur_val) == float(val)) if tipo == "bool" else (str(cur_val) == str(val))
    except (ValueError, TypeError):
        return False


def _print_match(p: pd.Series):
    wr   = p.get("wr_valid", np.nan)
    wr_s = f"{wr:.1%}" if isinstance(wr, float) and not np.isnan(wr) else "pendiente"
    tf_s = f" [{p['tf']}]" if pd.notna(p.get('tf')) else ""
    print(f"  [{p['direccion']}]{tf_s} {p['patron']}")
    print(f"    WR validado    : {wr_s}")
    print(f"    N historico    : {p.get('N', '?')}")
    print(f"    Mag media      : {p.get('mag_media', '?')}%  (std {p.get('mag_std','?')}%)")
    print(f"    MFE / MAE med  : {p.get('mfe_media','?')}% / {p.get('mae_media','?')}%")
    print(f"    Fib rebote     : {p.get('rebote_fib_frecuente', '?')}")
    print(f"    Fib agotamto   : {p.get('fib_agot_frecuente', '?')}")
    print(f"    Estado         : {p.get('estado', '?')}")
    print()


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
        matches = [p for _, p in validated.iterrows() if _match_pattern(p, cur)]

        if not matches:
            print("\n  Sin coincidencias con el estado actual.")
            # Mostrar cuantas variables del estado actual SI aparecen en patrones
            vars_en_patrones = set(validated["variables"].unique())
            vars_en_estado   = {k for k, v in cur.items() if v not in (None, False, 0, "NONE")}
            overlap = vars_en_patrones & vars_en_estado
            print(f"  Variables activas en estado: {len(vars_en_estado)}")
            print(f"  De ellas presentes en patrones: {len(overlap)}")
            if overlap:
                print(f"  Mas cercanas: {list(overlap)[:5]}")
        else:
            # Separar LONG y SHORT
            longs  = [p for p in matches if p["direccion"] == "LONG"]
            shorts = [p for p in matches if p["direccion"] == "SHORT"]

            print(f"\n  {len(matches)} PATRON(ES) ACTIVO(S)  "
                  f"[{len(longs)} LONG / {len(shorts)} SHORT]\n")

            if longs:
                print("  --- LONG ---")
                for p in sorted(longs,
                                key=lambda x: x.get("wr_valid") or 0,
                                reverse=True):
                    _print_match(p)

            if shorts:
                print("  --- SHORT ---")
                for p in sorted(shorts,
                                key=lambda x: x.get("wr_valid") or 0,
                                reverse=True):
                    _print_match(p)

    # Guardar estado actual en JSON
    path = os.path.join(RESULTS_DIR, f"{symbol}_current_state.json")
    with open(path, "w") as f:
        json.dump(
            {k: (None if isinstance(v, float) and np.isnan(v)
                 else (v.item() if hasattr(v, "item") else v))
             for k, v in cur.items()},
            f, indent=2, default=str
        )
    print(f"  Estado guardado: {path}")
