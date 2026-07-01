#!/usr/bin/env python3
"""
SISTEMA INSTITUCIONAL - PIPELINE NUEVO
No modifica ni reemplaza main.py existente.

Uso:
    python main2.py NVDA
    python main2.py NVDA 2023-01-01
    python main2.py NVDA 2023-01-01 2026-07-01
"""
import sys
import os
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))


def run(symbol, from_date="2024-01-01", to_date=None):
    to_date = to_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  SISTEMA INSTITUCIONAL --- {symbol}")
    print(f"  Periodo: {from_date}  ->  {to_date}")
    print(f"{'='*60}\n")

    print("[1/9] Descargando / cargando datos 1m...")
    from downloader import ensure_data
    df_1m = ensure_data(symbol, from_date, to_date)
    print(f"  {len(df_1m):,} velas  ({df_1m.index[0].date()} -> {df_1m.index[-1].date()})")

    print("\n[2/9] Resampleando a 8 TFs...")
    from resampler import build_all_timeframes
    tfs = build_all_timeframes(df_1m)
    for tf, df in tfs.items():
        print(f"  {tf}: {len(df):,} velas")

    print("\n[3/9] Calculando indicadores...")
    from indicators import add_all_indicators
    tfs_ind = {tf: add_all_indicators(df, tf_name=tf) for tf, df in tfs.items()}

    print("\n[4/9] Detectando estructura (swings / BOS / CHoCH)...")
    from modules.m4_structure import run_structure
    structure = run_structure(symbol, tfs_ind)

    print("\n[5/9] Detectando movimientos reales...")
    from modules.m5_movements import run_movements
    movements = run_movements(symbol, structure)
    if movements.empty:
        print("  AVISO: Sin movimientos detectados. Revisa el historico.")
        return

    print("\n[6/9] State matrix por movimiento...")
    from modules.m6_state_matrix import run_state_matrix
    df_start, df_end = run_state_matrix(symbol, movements, tfs_ind)

    print("\n[7/9] Calculando outcomes (MFE / MAE / Fibonacci)...")
    from modules.m7_outcomes import run_outcomes
    df_outcomes = run_outcomes(symbol, movements, df_1m)

    print("\n[8/9] Minando patrones (walk-forward)...")
    from modules.m8_pattern_miner import run_pattern_miner
    validated = run_pattern_miner(symbol, df_start, df_outcomes)

    print("\n[9/9] Estado actual vs patrones historicos...")
    from modules.m9_query_engine import run_query
    run_query(symbol, tfs_ind, validated)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  COMPLETADO en {elapsed:.1f}s")
    print(f"  Outputs -> data/processed/  y  data/results/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    if len(sys.argv) < 2:
        print("Uso: python main2.py SYMBOL [from_date] [to_date]")
        print("     python main2.py NVDA")
        print("     python main2.py NVDA 2023-01-01")
        sys.exit(1)
    sym = sys.argv[1].upper()
    fd  = sys.argv[2] if len(sys.argv) > 2 else "2024-01-01"
    td  = sys.argv[3] if len(sys.argv) > 3 else None
    run(sym, fd, td)
