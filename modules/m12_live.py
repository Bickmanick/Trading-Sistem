"""
MODULO 12 — MODO LIVE

Descarga las ultimas barras del activo (desde el ultimo dato
disponible hasta ahora), recalcula el state matrix con los
datos frescos y lanza el query engine para mostrar que patrones
coinciden con el estado actual del mercado.

Uso desde main.py:
    python main.py NVDA --live

Uso directo:
    python modules/m12_live.py NVDA

Requiere:
    - CSV historico ya existente en data/NVDA_1m.csv
    - Patrones validados en data/results/NVDA_validated_patterns.csv

Flujo:
    1. ensure_data(symbol, desde_ultimo_dato, hoy)  → actualiza CSV 1m
    2. build_all_timeframes + add_all_indicators
    3. build_state_matrix
    4. run_query(symbol, state_matrix, validated)
"""
import os
import sys

# Asegurar que la raiz del proyecto esta en el path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
from datetime import datetime, timezone
from config import DATA_DIR

RESULTS_DIR = os.path.join(DATA_DIR, "results")


def run_live(symbol: str) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ─── 1. Actualizar datos ───
    from downloader import ensure_data
    print(f"  [LIVE] Actualizando datos hasta {today}...")
    df_1m = ensure_data(symbol, from_date="2024-01-01", to_date=today)
    print(f"  [LIVE] {len(df_1m):,} velas disponibles "
          f"({df_1m.index[0].date()} -> {df_1m.index[-1].date()})")

    # ─── 2. Resample + Indicadores ───
    print("  [LIVE] Calculando indicadores...")
    from resampler import build_all_timeframes
    from indicators import add_all_indicators
    tfs     = build_all_timeframes(df_1m)
    tfs_ind = {tf: add_all_indicators(df, tf_name=tf)
               for tf, df in tfs.items()}

    # ─── 3. State Matrix ───
    print("  [LIVE] Construyendo state matrix...")
    from state_engine import build_state_matrix
    state_matrix = build_state_matrix(df_1m, tfs_ind)

    # ─── 4. Cargar patrones validados ───
    pat_path = os.path.join(RESULTS_DIR, f"{symbol}_validated_patterns.csv")
    if not os.path.exists(pat_path):
        print(f"  [LIVE] AVISO: No hay patrones validados en {pat_path}.")
        print(f"  [LIVE] Ejecuta primero: python main.py {symbol}")
        # Lanzar query sin patrones igualmente (muestra el estado actual)
        validated = pd.DataFrame()
    else:
        validated = pd.read_csv(pat_path)
        print(f"  [LIVE] {len(validated)} patrones validados cargados.")

    # ─── 5. Query Engine ───
    from modules.m9_query_engine import run_query
    run_query(symbol, state_matrix, validated)


if __name__ == "__main__":
    sym = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    run_live(sym)
