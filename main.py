#!/usr/bin/env python3
"""
SISTEMA DE ANALISIS MULTI-TIMEFRAME
Uso: python main.py SYMBOL
     python main.py NVDA
     python main.py AAPL 2024-01-01
"""
import sys, os, time
from datetime import datetime, timezone
import pandas as pd
from config import TF_NAMES, OUTPUT_DIR


def run(symbol: str, from_date="2024-01-01", to_date=None):
    # Si no se pasa fecha de fin, se usa hoy automáticamente
    if to_date is None:
        to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  TRADING SYSTEM — {symbol}")
    print(f"  Período: {from_date} → {to_date}")
    print(f"{'='*60}\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, symbol)
    os.makedirs(out, exist_ok=True)

    # ── 1. Datos ──────────────────────────────────────────────────
    print("[1/9] Cargando datos 1m...")
    from downloader import ensure_data
    df_1m = ensure_data(symbol, from_date, to_date)
    print(f"  {len(df_1m):,} velas 1m cargadas ({df_1m.index[0]} → {df_1m.index[-1]})")

    # ── 2. Resample ───────────────────────────────────────────────
    print("\n[2/9] Resampleando a 8 TFs...")
    from resampler import build_all_timeframes
    tfs = build_all_timeframes(df_1m)
    for tf, df in tfs.items():
        print(f"  {tf}: {len(df):,} velas")

    # ── 3. Indicadores ────────────────────────────────────────────
    print("\n[3/9] Calculando indicadores...")
    from indicators import add_all_indicators
    tfs_ind = {}
    for tf_name, df_tf in tfs.items():
        tfs_ind[tf_name] = add_all_indicators(df_tf, tf_name=tf_name)
        print(f"  {tf_name}: {len(df_tf.columns)} columnas → {len(tfs_ind[tf_name].columns)} con indicadores")

    # ── 4. State Matrix ───────────────────────────────────────────
    print("\n[4/9] Construyendo state_matrix...")
    from state_engine import build_state_matrix
    state = build_state_matrix(tfs_ind["1M"], tfs_ind)
    print(f"  State matrix: {state.shape[0]:,} filas × {state.shape[1]:,} columnas")
    state.to_csv(os.path.join(out, "state_matrix.csv"))

    # ── 5. Event Detection ────────────────────────────────────────
    print("\n[5/9] Detectando eventos...")
    from event_detector import detect_events
    state = detect_events(state)

    # ── 6. Alignment + Regime ─────────────────────────────────────
    print("\n[6/9] Calculando alineación y régimen...")
    from alignment_engine import compute_alignment
    state = compute_alignment(state)
    regime_dist = state["regime"].value_counts()
    print("  Distribución de regímenes:")
    for r, n in regime_dist.items():
        pct = n / len(state) * 100
        print(f"    {r:<25} {n:>8,} velas ({pct:.1f}%)")

    # ── 7. Thesis ─────────────────────────────────────────────────
    print("\n[7/9] Calculando calidad de tesis...")
    from thesis_engine import compute_thesis
    state = compute_thesis(state)
    tl = state["thesis_long"].value_counts()
    ts = state["thesis_short"].value_counts()
    print(f"  Thesis LONG:  {dict(tl)}")
    print(f"  Thesis SHORT: {dict(ts)}")

    # ── 8. Análisis Univariante + Pattern Detection ───────────────
    print("\n[8/9] Análisis univariante + detección de patrones...")
    from univariate_analysis import univariate_analysis
    edge_vars, df_uni = univariate_analysis(state)
    df_uni.to_csv(os.path.join(out, "univariate.csv"))

    from pattern_detector import detect_patterns
    patterns = detect_patterns(state, edge_vars=edge_vars, n_jobs=14)
    if not patterns.empty:
        patterns.to_csv(os.path.join(out, "patterns.csv"))
        print(f"  Guardado: patterns.csv ({len(patterns):,} filas)")

    # ── 9. Trade Cycle + Estadísticas ─────────────────────────────
    print("\n[9/9] Simulando ciclo de trades...")
    from trade_cycle import run_trade_cycle
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
        print(f"  MAE medio:       {trades.get('mae_pct', pd.Series([0])).mean():.3f}%")
        print(f"  MFE medio:       {trades.get('mfe_pct', pd.Series([0])).mean():.3f}%")

        from statistics import compute_statistics, save_statistics
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
        print("Uso: python main.py SYMBOL [from_date] [to_date]")
        print("Ej:  python main.py NVDA 2024-01-01")
        print("     python main.py NVDA  (descarga hasta hoy automáticamente)")
        sys.exit(1)
    sym = sys.argv[1].upper()
    fd  = sys.argv[2] if len(sys.argv) > 2 else "2024-01-01"
    td  = sys.argv[3] if len(sys.argv) > 3 else None  # None = hoy automático
    run(sym, fd, td)
