#!/usr/bin/env python3
"""
DIAGNOSTICO - Inspecciona los movimientos generados por m5
y calcula manualmente fib_agotamiento para los primeros N movimientos
para verificar si precio_extremo es correcto.

Uso:
    python diag.py NVDA
    python diag.py NVDA 20   <- muestra 20 movimientos
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))

from config import DATA_DIR

PROCESSED = os.path.join(DATA_DIR, "processed")


def run_diag(symbol: str, n: int = 30):
    # 1. Cargar movements linked
    path_mov = os.path.join(PROCESSED, f"{symbol}_movements_linked.csv")
    if not os.path.exists(path_mov):
        print(f"No existe {path_mov} - ejecuta primero main.py {symbol}")
        return
    movs = pd.read_csv(path_mov)
    movs["timestamp_inicio"] = pd.to_datetime(movs["timestamp_inicio"], utc=True)
    movs["timestamp_fin"]    = pd.to_datetime(movs["timestamp_fin"],    utc=True)

    # 2. Cargar outcomes
    path_out = os.path.join(PROCESSED, f"{symbol}_outcomes.csv")
    if not os.path.exists(path_out):
        print(f"No existe {path_out}")
        return
    outs = pd.read_csv(path_out)

    # 3. Cargar 1M
    path_1m = os.path.join(DATA_DIR, "raw", f"{symbol}_1M.parquet")
    if not os.path.exists(path_1m):
        path_1m = os.path.join(DATA_DIR, "raw", f"{symbol}_1m.parquet")
    if not os.path.exists(path_1m):
        print(f"No encuentro {symbol}_1M.parquet en data/raw/")
        df_1m = None
    else:
        df_1m = pd.read_parquet(path_1m)
        df_1m.index = pd.to_datetime(df_1m.index, utc=True)
        df_1m = df_1m.sort_index()
        print(f"1M cargado: {len(df_1m):,} velas")

    print(f"\n{'='*70}")
    print(f"DIAGNOSTICO {symbol} - primeros {n} movimientos cerrados")
    print(f"{'='*70}")

    # Estadisticas globales de outcomes
    if "fib_agotamiento" in outs.columns:
        dist = outs["fib_agotamiento"].dropna().value_counts(normalize=True).sort_index()
        print(f"\nDistribucion fib_agotamiento en outcomes ({len(outs)} movs):")
        for v, pct in dist.items():
            bar = '#' * int(pct * 40)
            print(f"  {v:.3f} : {pct:5.1%}  {bar}")
    if "rebote_fib" in outs.columns:
        dist2 = outs["rebote_fib"].dropna().value_counts(normalize=True).sort_index()
        print(f"\nDistribucion rebote_fib en outcomes:")
        for v, pct in dist2.items():
            bar = '#' * int(pct * 40)
            print(f"  {v:.3f} : {pct:5.1%}  {bar}")

    print(f"\nBalance LONG/SHORT en movimientos:")
    bal = movs["direccion"].value_counts()
    for d, c in bal.items():
        print(f"  {d}: {c:,} ({c/len(movs)*100:.1f}%)")

    print(f"\nBalance por TF:")
    for tf in ["1M","5M","15M","30M","1H","4H","1D","1W"]:
        sub = movs[movs["tf"] == tf]
        if sub.empty:
            continue
        nl = (sub["direccion"]=="LONG").sum()
        ns = (sub["direccion"]=="SHORT").sum()
        print(f"  {tf:4s}: LONG={nl:5d} SHORT={ns:5d}  ratio={nl/(ns if ns else 1):.2f}")

    print(f"\n--- Muestra de {n} movimientos (4H y 1H) con calculo manual de fib ---")
    sample = movs[movs["tf"].isin(["4H","1H"])].dropna(subset=["timestamp_fin"]).head(n)

    for _, mov in sample.iterrows():
        pi = mov["precio_inicio"]
        pe = mov["precio_extremo"]
        pf = mov["precio_fin"]
        rango = abs(pe - pi)
        print(f"\n  mov_id={mov['mov_id']} tf={mov['tf']} dir={mov['direccion']}")
        print(f"  precio_inicio={pi:.4f}  precio_extremo={pe:.4f}  precio_fin={pf:.4f}")
        print(f"  rango={rango:.4f}  magnitud={mov.get('magnitud_pct',0):.3f}%")
        print(f"  ts_inicio={mov['timestamp_inicio']}  ts_fin={mov['timestamp_fin']}")

        if rango == 0:
            print(f"  !! RANGO=0: precio_extremo == precio_inicio (bug m5)")
            continue
        if pd.isna(pf):
            print(f"  !! precio_fin=NaN")
            continue

        # Calcular fib de agotamiento manual con df_1m
        if df_1m is not None:
            ts_ini = mov["timestamp_inicio"]
            ts_fin_m = mov["timestamp_fin"]
            w = df_1m[(df_1m.index >= ts_ini) & (df_1m.index <= ts_fin_m)]
            if not w.empty:
                if mov["direccion"] == "LONG":
                    min_int = w["low"].min()
                    ret_int = pe - min_int
                else:
                    max_int = w["high"].max()
                    ret_int = max_int - pe
                ratio_int = max(0.0, min(1.0, ret_int / rango))
                FIB = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
                fib_calc = min(FIB, key=lambda f: abs(f - ratio_int))
                print(f"  fib_agotamiento CALCULADO = {fib_calc:.3f} (ratio={ratio_int:.4f})")
                if len(w) < 3:
                    print(f"  AVISO: solo {len(w)} velas 1M en ventana (movimiento muy corto)")
            else:
                print(f"  !! Sin velas 1M en ventana [{ts_ini}, {ts_fin_m}]")

        # Buscar en outcomes
        out_row = outs[outs["mov_id"] == mov["mov_id"]]
        if not out_row.empty:
            print(f"  fib_agotamiento EN CSV = {out_row.iloc[0].get('fib_agotamiento', 'N/A')}")
            print(f"  rebote_fib EN CSV      = {out_row.iloc[0].get('rebote_fib', 'N/A')}")
        else:
            print(f"  mov_id no encontrado en outcomes")

    print(f"\n{'='*70}")
    print(f"Si fib_agotamiento CALCULADO != EN CSV -> bug en m7")
    print(f"Si rango=0 sistematicamente -> bug en m5 (precio_extremo=precio_inicio)")
    print(f"Si ratio siempre ~0 -> precio_extremo se guarda como precio de cierre")
    print(f"{'='*70}")


if __name__ == "__main__":
    sym = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    n   = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    run_diag(sym, n)
