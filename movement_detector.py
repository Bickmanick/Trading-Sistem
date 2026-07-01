"""
movement_detector.py v2
Detecta impulsos reales del precio usando MACD de 5M (menos ruido que 1M).
Filtra movimientos < 1% o < 30 barras de 1M.
"""
import pandas as pd
import numpy as np

def detect_movements(sm: pd.DataFrame, min_pct: float = 1.0, min_bars: int = 30) -> pd.DataFrame:
    """
    Motor: histograma MACD de 5M.
    Inicio LONG:  hist_5M cruza de negativo a positivo
    Inicio SHORT: hist_5M cruza de positivo a negativo
    Fin:          hist_5M vuelve al lado contrario
                  O estocástico 5M entra en zona extrema + hist_5M shrinka >= 5 barras
    """
    prices   = sm["close"].values
    # usar MACD de 5M como motor principal
    hist_col = "5M_macd_hist" if "5M_macd_hist" in sm.columns else "1M_macd_hist"
    hist     = sm[hist_col].fillna(0).values
    n        = len(prices)

    # columnas auxiliares para agotamiento
    stoch_ob_5m  = sm.get("5M_stoch_ob",            pd.Series(np.zeros(n))).fillna(0).values
    stoch_os_5m  = sm.get("5M_stoch_os",            pd.Series(np.zeros(n))).fillna(0).values
    fall_5m      = sm.get("5M_macd_hist_bars_falling", pd.Series(np.zeros(n))).fillna(0).values
    rise_5m      = sm.get("5M_macd_hist_bars_rising",  pd.Series(np.zeros(n))).fillna(0).values

    # columnas auxiliares para agotamiento en TFs mayores
    stoch_ob_1h  = sm.get("1H_stoch_ob",  pd.Series(np.zeros(n))).fillna(0).values
    stoch_os_1h  = sm.get("1H_stoch_os",  pd.Series(np.zeros(n))).fillna(0).values

    movements = []
    i = 1

    while i < n - min_bars:
        direction = None

        # cruce MACD 5M: <=0 → >0 (LONG)
        if hist[i-1] <= 0 and hist[i] > 0:
            direction = "LONG"
        # cruce MACD 5M: >=0 → <0 (SHORT)
        elif hist[i-1] >= 0 and hist[i] < 0:
            direction = "SHORT"

        if direction is None:
            i += 1
            continue

        start = i
        end   = start + min_bars  # mínimo

        # buscar fin del movimiento
        while end < n:
            if direction == "LONG":
                # agotamiento: hist vuelve negativo
                if hist[end] <= 0:
                    break
                # O: stoch OB en 5M + shrink >= 5 barras
                if stoch_ob_5m[end] and fall_5m[end] >= 5:
                    break
                # O: stoch OB en 1H (contexto mayor confirmando agotamiento)
                if stoch_ob_1h[end] and fall_5m[end] >= 3:
                    break
            else:  # SHORT
                if hist[end] >= 0:
                    break
                if stoch_os_5m[end] and rise_5m[end] >= 5:
                    break
                if stoch_os_1h[end] and rise_5m[end] >= 3:
                    break
            end += 1

        end = min(end, n - 1)

        ep  = prices[start]
        xp  = prices[end]
        if ep <= 0 or np.isnan(ep) or np.isnan(xp):
            i = end + 1
            continue

        pct      = (xp - ep) / ep * 100
        duration = end - start

        # filtrar movimientos insignificantes
        if abs(pct) >= min_pct and duration >= min_bars:
            ts_start = sm["timestamp"].iloc[start] if "timestamp" in sm.columns else str(start)
            ts_end   = sm["timestamp"].iloc[end]   if "timestamp" in sm.columns else str(end)

            # max favorable excursion (máximo a favor durante el movimiento)
            segment = prices[start:end+1]
            if direction == "LONG":
                mfe = (np.max(segment) - ep) / ep * 100
            else:
                mfe = (ep - np.min(segment)) / ep * 100

            movements.append({
                "id":            len(movements),
                "direction":     direction,
                "bar_start":     start,
                "bar_end":       end,
                "ts_start":      ts_start,
                "ts_end":        ts_end,
                "price_start":   round(ep, 4),
                "price_end":     round(xp, 4),
                "pct_move":      round(pct, 3),
                "mfe_pct":       round(mfe, 3),   # max a favor
                "duration_bars": duration,
                "duration_hrs":  round(duration / 60, 1),
            })

        i = end + 1  # saltar al siguiente

    df = pd.DataFrame(movements)
    print(f"  Movimientos detectados: {len(df)}")
    if len(df):
        long_  = df[df.direction=="LONG"]
        short_ = df[df.direction=="SHORT"]
        if len(long_):
            print(f"    LONG:  {len(long_):3d} | med_pct={long_['pct_move'].median():+.2f}% | "
                  f"max_mfe={long_['mfe_pct'].median():+.2f}% | med_dur={long_['duration_hrs'].median():.1f}H")
        if len(short_):
            print(f"    SHORT: {len(short_):3d} | med_pct={short_['pct_move'].median():+.2f}% | "
                  f"max_mfe={short_['mfe_pct'].median():+.2f}% | med_dur={short_['duration_hrs'].median():.1f}H")
    return df
