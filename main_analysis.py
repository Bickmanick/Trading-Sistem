"""
main_analysis.py v2
"""
import pandas as pd
import numpy as np
from pathlib import Path

from movement_detector import detect_movements
from state_snapshot    import snapshot_states
from pattern_stats     import compute_pattern_stats

SYMBOL     = "NVDA"
OUTPUT_DIR = Path("output") / SYMBOL
MIN_PCT    = 1.0   # movimiento mínimo 1%
MIN_BARS   = 30    # mínimo 30 barras de 1M (30 minutos)

print("=" * 70)
print(f"SISTEMA DE ANÁLISIS DE MOVIMIENTOS v2 — {SYMBOL}")
print("=" * 70)
print()

print("1. Cargando state_matrix.csv...")
sm = pd.read_csv(OUTPUT_DIR / "state_matrix.csv")
print(f"   Filas: {len(sm):,} | Columnas: {len(sm.columns)}")
print(f"   Rango: {sm['timestamp'].iloc[0]} → {sm['timestamp'].iloc[-1]}")

print()
print("2. Detectando movimientos (motor: MACD 5M, min 1%, min 30 barras)...")
movements = detect_movements(sm, min_pct=MIN_PCT, min_bars=MIN_BARS)
movements.to_csv(OUTPUT_DIR / "movements.csv", index=False)

print()
print("3. Capturando estados en inicio y fin de cada movimiento...")
entry_states, exit_states = snapshot_states(sm, movements)
entry_states.to_csv(OUTPUT_DIR / "entry_states.csv", index=False)
exit_states.to_csv(OUTPUT_DIR  / "exit_states.csv",  index=False)

print()
print("4. Estadísticas por patrón...")
print()
print("   [PATRONES DE ENTRADA]")
entry_patterns = compute_pattern_stats(entry_states, label="entry")
if len(entry_patterns):
    entry_patterns.to_csv(OUTPUT_DIR / "pattern_stats_entry.csv", index=False)

print()
print("   [PATRONES DE AGOTAMIENTO]")
exit_patterns = compute_pattern_stats(exit_states, label="exit")
if len(exit_patterns):
    exit_patterns.to_csv(OUTPUT_DIR / "pattern_stats_exit.csv", index=False)

print()
print("=" * 70)
print("TOP 20 PATRONES DE ENTRADA")
print("=" * 70)
if len(entry_patterns):
    for i, r in entry_patterns.head(20).iterrows():
        print(f"  #{i+1:2d} [{r.dominant_dir}] n={r.n:3d} | alin={r.pct_aligned:5.1f}% | "
              f"mov_med={r.pct_move_med:+.2f}% | mfe_med={r.mfe_med:+.2f}% | dur={r.dur_hrs_med:.1f}H")
        parts = [p for p in r.pattern.split("|") if p and p != "NO_SIGNAL"]
        for chunk in [parts[k:k+3] for k in range(0, len(parts), 3)]:
            print(f"       {" | ".join(chunk)}")
        print()

print()
print("=" * 70)
print("TOP 20 PATRONES DE AGOTAMIENTO")
print("=" * 70)
if len(exit_patterns):
    for i, r in exit_patterns.head(20).iterrows():
        print(f"  #{i+1:2d} [{r.dominant_dir}→AGOT] n={r.n:3d} | alin={r.pct_aligned:5.1f}% | "
              f"mov_med={r.pct_move_med:+.2f}% | dur={r.dur_hrs_med:.1f}H")
        parts = [p for p in r.pattern.split("|") if p and p != "NO_SIGNAL"]
        for chunk in [parts[k:k+3] for k in range(0, len(parts), 3)]:
            print(f"       {" | ".join(chunk)}")
        print()

print()
print("Archivos generados:")
for f in ["movements.csv","entry_states.csv","exit_states.csv",
          "pattern_stats_entry.csv","pattern_stats_exit.csv"]:
    p = OUTPUT_DIR / f
    if p.exists():
        size = p.stat().st_size // 1024
        print(f"  ✓ output/{SYMBOL}/{f}  ({size} KB)")
