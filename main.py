#!/usr/bin/env python3
"""
SISTEMA INSTITUCIONAL — Pipeline principal

Uso:
    python main.py NVDA
    python main.py NVDA 2024-01-01
    python main.py NVDA 2024-01-01 2026-07-01
    python main.py NVDA 2024-01-01 2026-07-01 --live

Outputs:
    data/processed/   -> CSVs intermedios
    data/results/     -> patrones validados, estado actual, informe
    data/results/{symbol}_run.log
"""
import sys
import os
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))


class _Tee:
    """Escribe en stdout Y en fichero de log simultaneamente."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
    def isatty(self):
        return False


def run(symbol: str, from_date: str = "2024-01-01",
        to_date: str = None, live: bool = False):
    to_date = to_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    t0 = time.time()

    from config import DATA_DIR
    results_dir = os.path.join(DATA_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"{symbol}_run.log")
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log_file)
    print(f"Log guardado en: {log_path}")

    try:
        print(f"\n{'='*60}")
        print(f"  SISTEMA INSTITUCIONAL --- {symbol}")
        print(f"  Periodo: {from_date}  ->  {to_date}")
        if live:
            print(f"  MODO: LIVE")
        print(f"{'='*60}\n")

        # --- MODO LIVE: solo descarga + query ---
        if live:
            print("[LIVE] Descargando ultima barra y ejecutando query...")
            from modules.m12_live import run_live
            run_live(symbol)
            elapsed = time.time() - t0
            print(f"\n  LIVE completado en {elapsed:.1f}s")
            return

        # ─── PASO 1: Datos 1m ───
        print("[1/9] Descargando / cargando datos 1m...")
        from downloader import ensure_data
        df_1m = ensure_data(symbol, from_date, to_date)
        print(f"  {len(df_1m):,} velas  "
              f"({df_1m.index[0].date()} -> {df_1m.index[-1].date()})")

        # ─── PASO 2: Resample ───
        print("\n[2/9] Resampleando a 8 TFs...")
        from resampler import build_all_timeframes
        tfs = build_all_timeframes(df_1m)
        for tf, df in tfs.items():
            print(f"  {tf}: {len(df):,} velas")

        # ─── PASO 3: Indicadores ───
        print("\n[3/9] Calculando indicadores...")
        from indicators import add_all_indicators
        tfs_ind = {tf: add_all_indicators(df, tf_name=tf)
                   for tf, df in tfs.items()}

        # ─── PASO 4: State Matrix completo ───
        print("\n[4/9] Construyendo state matrix...")
        from state_engine import build_state_matrix
        state_matrix = build_state_matrix(df_1m, tfs_ind)
        print(f"  State matrix: {len(state_matrix):,} filas "
              f"x {len(state_matrix.columns)} columnas")

        # Guardar state matrix
        from config import DATA_DIR as _DD
        _proc = os.path.join(_DD, "processed")
        os.makedirs(_proc, exist_ok=True)
        sm_path = os.path.join(_proc, f"{symbol}_state_matrix.csv")
        state_matrix.to_csv(sm_path)
        print(f"  Guardado: {sm_path}")

        # ─── PASO 5: Estructura ───
        print("\n[5/9] Detectando estructura (swings / BOS / CHoCH)...")
        from modules.m4_structure import run_structure
        structure = run_structure(symbol, tfs_ind)

        # ─── PASO 6: Movimientos ───
        print("\n[6/9] Detectando movimientos reales...")
        from modules.m5_movements import run_movements
        movements = run_movements(symbol, structure)
        if movements.empty:
            print("  AVISO: Sin movimientos detectados.")
            return
        print(f"  {len(movements)} movimientos vinculados.")

        # ─── PASO 7: Estado por movimiento ───
        print("\n[7/9] State matrix por movimiento...")
        from modules.m6_state_matrix import run_state_matrix
        df_start, df_end = run_state_matrix(symbol, movements, state_matrix)

        # ─── PASO 8: Outcomes ───
        print("\n[8/9] Calculando outcomes (MFE/MAE/Fib)...")
        from modules.m7_outcomes import run_outcomes
        df_outcomes = run_outcomes(symbol, movements, df_1m)

        # ─── PASO 9a: Pattern Miner ───
        print("\n[9/9a] Minando patrones (walk-forward)...")
        from modules.m8_pattern_miner import run_pattern_miner
        validated = run_pattern_miner(symbol, df_start, df_outcomes)

        # ─── PASO 9b: Walk-forward extendido ───
        print("\n[9/9b] Validacion walk-forward extendida (M10)...")
        from modules.m10_walkforward import run_walkforward
        run_walkforward(symbol, df_start, df_outcomes, validated)

        # ─── PASO 9c: Query Engine ───
        print("\n[9/9c] Estado actual vs patrones historicos (M9)...")
        from modules.m9_query_engine import run_query
        run_query(symbol, state_matrix, validated)

        # ─── PASO 9d: Informe ───
        print("\n[9/9d] Generando informe (M11)...")
        from modules.m11_report import run_report
        run_report(symbol, validated, df_outcomes)

        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"  COMPLETADO en {elapsed:.1f}s")
        print(f"  Outputs -> data/processed/  y  data/results/")
        print(f"  Log completo -> {log_path}")
        print(f"{'='*60}\n")

    except Exception as e:
        import traceback
        print(f"\n  ERROR: {e}")
        traceback.print_exc()
        raise
    finally:
        sys.stdout = original_stdout
        log_file.close()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    if len(sys.argv) < 2:
        print("Uso: python main.py SYMBOL [from_date] [to_date] [--live]")
        print("     python main.py NVDA")
        print("     python main.py NVDA 2024-01-01")
        print("     python main.py NVDA 2024-01-01 2026-07-07 --live")
        sys.exit(1)

    sym   = sys.argv[1].upper()
    fd    = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "2024-01-01"
    td    = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else None
    _live = "--live" in sys.argv

    run(sym, fd, td, live=_live)
