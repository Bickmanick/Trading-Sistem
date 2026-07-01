"""
MODULO 4 — DETECTOR DE ESTRUCTURA
Responsabilidad: detectar swings reales y la estructura del mercado
(HH, HL, LH, LL, BOS, CHoCH) en cada TF sin lookahead.

Outputs por TF:
    data/processed/{symbol}_structure_{tf}.csv
    columnas: timestamp, swing_type, swing_price, estructura, bos_event, choch_event
"""

import os
import pandas as pd
import numpy as np
from config import TIMEFRAMES, DATA_DIR

# Barras de confirmacion necesarias por TF (sin lookahead)
SWING_N = {
    "1W":  8,
    "1D": 10,
    "4H": 10,
    "1H": 10,
    "30M": 10,
    "15M":  8,
    "5M":   8,
    "1M":   5,
}

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def _detect_swings(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Detecta swing highs y swing lows confirmados sin lookahead.
    Un swing high en barra i es valido si:
      - high[i] es el maximo de [i-n .. i-1]  (solo mira atras)
      - han pasado al menos n barras desde el ultimo swing high
    Analogamente para swing low.
    """
    highs = df["high"].values
    lows  = df["low"].values
    n_bars = len(df)

    swing_type  = ["NONE"] * n_bars
    swing_price = [np.nan]  * n_bars

    last_sh_idx = -999
    last_sl_idx = -999

    for i in range(n, n_bars):
        window_h = highs[i - n: i]
        window_l = lows[i  - n: i]

        # Swing High: el high de hace n barras es maximo de las n anteriores
        candidate_h = highs[i - n]
        if candidate_h == window_h.max() and (i - n) - last_sh_idx >= n:
            swing_type[i - n]  = "SH"
            swing_price[i - n] = candidate_h
            last_sh_idx        = i - n

        # Swing Low: el low de hace n barras es minimo de las n anteriores
        candidate_l = lows[i - n]
        if candidate_l == window_l.min() and (i - n) - last_sl_idx >= n:
            swing_type[i - n]  = "SL"
            swing_price[i - n] = candidate_l
            last_sl_idx        = i - n

    df = df.copy()
    df["swing_type"]  = swing_type
    df["swing_price"] = swing_price
    return df


def _classify_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Con los swings confirmados, clasifica la estructura para cada barra:
    ALCISTA  → HH + HL
    BAJISTA  → LH + LL
    RANGO    → sin direccion clara
    """
    swings = df[df["swing_type"].isin(["SH", "SL"])].copy()

    estructura   = ["RANGO"] * len(df)
    bos_event    = ["NONE"]  * len(df)
    choch_event  = ["NONE"]  * len(df)

    sh_prices = []
    sl_prices = []

    swing_idx_map = {}
    for pos, (idx, row) in enumerate(df.iterrows()):
        if row["swing_type"] == "SH":
            sh_prices.append(row["swing_price"])
        elif row["swing_type"] == "SL":
            sl_prices.append(row["swing_price"])

        # Necesitamos al menos 2 swings de cada tipo para clasificar
        if len(sh_prices) >= 2 and len(sl_prices) >= 2:
            last_sh   = sh_prices[-1]
            prev_sh   = sh_prices[-2]
            last_sl   = sl_prices[-1]
            prev_sl   = sl_prices[-2]

            hh = last_sh > prev_sh   # Higher High
            hl = last_sl > prev_sl   # Higher Low
            lh = last_sh < prev_sh   # Lower High
            ll = last_sl < prev_sl   # Lower Low

            if hh and hl:
                estructura[pos] = "ALCISTA"
            elif lh and ll:
                estructura[pos] = "BAJISTA"
            else:
                estructura[pos] = "RANGO"

            close = df["close"].iloc[pos]

            # BOS — confirmacion de continuacion
            if estructura[pos] == "ALCISTA" and close > prev_sh:
                bos_event[pos] = "BOS_BULL"
            elif estructura[pos] == "BAJISTA" and close < prev_sl:
                bos_event[pos] = "BOS_BEAR"

            # CHoCH — cambio de caracter (senal de giro)
            if estructura[pos] == "ALCISTA" and close < last_sl:
                choch_event[pos] = "CHOCH_BAJISTA"
            elif estructura[pos] == "BAJISTA" and close > last_sh:
                choch_event[pos] = "CHOCH_ALCISTA"

    df = df.copy()
    df["estructura"]  = estructura
    df["bos_event"]   = bos_event
    df["choch_event"] = choch_event
    return df


def run_structure(symbol: str, tfs_data: dict) -> dict:
    """
    Punto de entrada principal.
    tfs_data: dict {tf_name: DataFrame con OHLCV + indicadores}
    Devuelve dict {tf_name: DataFrame con estructura anadida}
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    results = {}

    for tf_name, df in tfs_data.items():
        n = SWING_N.get(tf_name, 10)
        print(f"  [{tf_name}] Detectando swings (N={n})...")

        df_s = _detect_swings(df, n)
        df_s = _classify_structure(df_s)

        n_sh    = (df_s["swing_type"] == "SH").sum()
        n_sl    = (df_s["swing_type"] == "SL").sum()
        n_bos   = df_s["bos_event"].ne("NONE").sum()
        n_choch = df_s["choch_event"].ne("NONE").sum()

        print(f"  [{tf_name}] SH={n_sh} SL={n_sl} BOS={n_bos} CHoCH={n_choch}")

        # Guardar CSV
        out_path = os.path.join(PROCESSED_DIR, f"{symbol}_structure_{tf_name}.csv")
        cols = ["swing_type", "swing_price", "estructura", "bos_event", "choch_event"]
        df_s[cols].to_csv(out_path)
        print(f"  [{tf_name}] Guardado: {out_path}")

        results[tf_name] = df_s

    return results


if __name__ == "__main__":
    # Test rapido: carga datos de NVDA y ejecuta estructura en 1H
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from downloader import ensure_data
    from resampler import build_all_timeframes
    from indicators import add_all_indicators

    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    print(f"Test m4_structure — {symbol}")
    df_1m = ensure_data(symbol)
    tfs   = build_all_timeframes(df_1m)
    tfs_ind = {tf: add_all_indicators(df, tf_name=tf) for tf, df in tfs.items()}
    run_structure(symbol, tfs_ind)
    print("Done.")
