#!/usr/bin/env python3
"""
SISTEMA DE ANALISIS MULTI-TIMEFRAME (LEGACY)
Uso: python legacy/main_legacy.py SYMBOL
     python legacy/main_legacy.py NVDA
     python legacy/main_legacy.py AAPL 2024-01-01

NOTA: Este es el sistema original basado en regimen/thesis/KNN.
El sistema activo es main.py (institucional CHoCH).
"""
import sys, os, time
from datetime import datetime, timezone
import pandas as pd

# Asegurar que encuentra los modulos en la raiz
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TF_NAMES, OUTPUT_DIR


def run(symbol: str, from_date="2024-01-01", to_date=None):
    if to_date is None:
        to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  TRADING SYSTEM LEGACY — {symbol}")
    print(f"  Periodo: {from_date} -> {to_date}")
    print(f"{'='*60}\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, symbol)
    os.makedirs(out, exist_ok=True)

    print("[1/9] Cargando datos 1m...")
    from downloader import ensure_data
    df_1m = ensure_data(symbol, from_date, to_date)
    print(f"  {len(df_1m):,} velas 1m cargadas ({df_1m.index[0]} -> {df_1m.index[-1]})")

    print("\n[2/9] Resampleando a 8 TFs...")
    from resampler import build_all_timeframes
    tfs = build_all_timeframes(df_1m)
    for tf, df in tfs.items():
        print(f"  {tf}: {len(df):,} velas")

    print("\n[3/9] Calculando indicadores...")
    from indicators import add_all_indicators
    tfs_ind = {}
    for tf_name, df_tf in tfs.items():
        tfs_ind[tf_name] = add_all_indicators(df_tf, tf_name=tf_name)
        print(f"  {tf_name}: {len(df_tf.columns)} columnas -> {len(tfs_ind[tf_name].columns)} con indicadores")

    print("\n[4/9] Construyendo state_matrix...")
    from state_engine import build_state_matrix
    state = build_state_matrix(tfs_ind["1M"], tfs_ind)
    print(f"  State matrix: {state.shape[0]:,} filas x {state.shape[1]:,} columnas")
    state.to_csv(os.path.join(out, "state_matrix.csv"))

    print("\n[5/9] Detectando eventos...")
    from legacy.event_detector import detect_events
    state = detect_events(state)

    print("\n[6/9] Calculando alineacion y regimen...")
    from legacy.alignment_engine import compute_alignment
    state = compute_alignment(state)
    regime_dist = state["regime"].value_counts()
    print("  Distribucion de regimenes:")
    for r, n in regime_dist.items():
        pct = n / len(state) * 100
        print(f"    {r:<25} {n:>8,} velas ({pct:.1f}%)")

    print("\n[7/9] Calculando calidad de tesis...")
    from legacy.thesis_engine import compute_thesis
    state = compute_thesis(state)
    tl = state["thesis_long"].value_counts()
    ts = state["thesis_short"].value_counts()
    print(f"  Thesis LONG:  {dict(tl)}")
    print(f"  Thesis SHORT: {dict(ts)}")

    print("\n[8/9] Analisis univariante + deteccion de patrones...")
    from legacy.univariate_analysis import univariate_analysis
    edge_vars, df_uni = univariate_analysis(state)
    df_uni.to_csv(os.path.join(out, "univariate.csv"))

    from legacy.pattern_detector import detect_patterns
    patterns = detect_patterns(state, edge_vars=edge_vars, n_jobs=14)
    if not patterns.empty:
        patterns.to_csv(os.path.join(out, "patterns.csv"))
        print(f"  Guardado: patterns.csv ({len(patterns):,} filas)")

    print("\n[9/9] Simulando ciclo de trades...")
    from legacy.trade_cycle import run_trade_cycle
    trades = run_trade_cycle(state, patterns)
    if not trades.empty:
        trades.to_csv(os.path.join(out, "trades.csv"), index=False)
        wr      = (trades["pnl_pct"] > 0).mean()
        avg_pnl = trades["pnl_pct"].mean()
        n_t     = len(trades)
        print(f"\n  RESUMEN DE TRADES ({symbol}):")
        print(f"  Total trades:    {n_t}")
        print(f"  Win rate:        {wr:.1%}")
        print(f"  PnL medio:       {avg_pnl:.3f}%")

        from legacy.statistics import compute_statistics, save_statistics
        stats = compute_statistics(trades, patterns)
        save_statistics(stats, OUTPUT_DIR, symbol)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  COMPLETADO en {elapsed:.1f}s — outputs en output/{symbol}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    if len(sys.argv) < 2:
        print("Uso: python legacy/main_legacy.py SYMBOL [from_date] [to_date]")
        sys.exit(1)
    sym = sys.argv[1].upper()
    fd  = sys.argv[2] if len(sys.argv) > 2 else "2024-01-01"
    td  = sys.argv[3] if len(sys.argv) > 3 else None
    run(sym, fd, td)
