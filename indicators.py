import numpy as np
import pandas as pd
from config import (
    MACD_FAST, MACD_SLOW, MACD_SIGNAL, STOCH_K, STOCH_D, STOCH_SMOOTH,
    STOCH_OB, STOCH_OS, EMA_FAST, EMA_MID, SMA_SLOW, SMA_MACRO,
    FIB_LEVELS, FIB_ZONE_PCT, FIB_SWING_BARS,
)

def bars_consecutive(s):
    b = s.astype(int)
    groups = (b != b.shift()).cumsum()
    counts = b.groupby(groups).cumcount() + 1
    return (counts * b).astype(int)

def add_macd(df, p=""):
    c = df["close"]
    ef = c.ewm(span=MACD_FAST, adjust=False).mean()
    es = c.ewm(span=MACD_SLOW, adjust=False).mean()
    line = ef - es
    sig  = line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    hist = line - sig
    d    = hist.diff()
    df[p+"macd_hist"]              = hist
    df[p+"macd_line"]              = line
    df[p+"macd_signal_line"]       = sig
    df[p+"macd_hist_delta"]        = d
    df[p+"macd_above_zero"]        = (hist > 0).astype(int)
    df[p+"macd_line_above_zero"]   = (line > 0).astype(int)
    df[p+"macd_above_sig"]         = (line > sig).astype(int)
    df[p+"macd_e1_bull"]           = ((d > 0) & (d.shift(1) <= 0)).astype(int)
    df[p+"macd_e1_bear"]           = ((d < 0) & (d.shift(1) >= 0)).astype(int)
    prev_a = line.shift(1) > sig.shift(1)
    curr_a = line > sig
    df[p+"macd_e2_bull"]           = ((curr_a) & (~prev_a)).astype(int)
    df[p+"macd_e2_bear"]           = ((~curr_a) & (prev_a)).astype(int)
    df[p+"macd_hist_bars_rising"]  = bars_consecutive(d > 0)
    df[p+"macd_hist_bars_falling"] = bars_consecutive(d < 0)
    df[p+"macd_line_dist"]         = line - sig
    return df

def add_stoch(df, p=""):
    low14  = df["low"].rolling(STOCH_K).min()
    high14 = df["high"].rolling(STOCH_K).max()
    rng    = (high14 - low14).replace(0, np.nan)
    K = (100 * (df["close"] - low14) / rng).rolling(STOCH_D).mean()
    D = K.rolling(STOCH_SMOOTH).mean()
    df[p+"stoch_K"]          = K
    df[p+"stoch_D"]          = D
    df[p+"stoch_K_delta"]    = K.diff()
    df[p+"stoch_KD_dist"]    = K - D
    kda = (K > D).astype(int)
    df[p+"stoch_KD_above"]   = kda
    df[p+"stoch_ob"]         = (K > STOCH_OB).astype(int)
    df[p+"stoch_os"]         = (K < STOCH_OS).astype(int)
    df[p+"stoch_bars_extreme"] = bars_consecutive((K > STOCH_OB) | (K < STOCH_OS))
    ros = (K.shift(0)<STOCH_OS)|(K.shift(1)<STOCH_OS)|(K.shift(2)<STOCH_OS)
    rob = (K.shift(0)>STOCH_OB)|(K.shift(1)>STOCH_OB)|(K.shift(2)>STOCH_OB)
    cup = (kda==1) & (kda.shift(1)==0)
    cdn = (kda==0) & (kda.shift(1)==1)
    df[p+"stoch_A_bull"] = (cup & ros).astype(int)
    df[p+"stoch_A_bear"] = (cdn & rob).astype(int)
    mid = (K >= 30) & (K <= 70)
    df[p+"stoch_B_bull"] = (cup & mid).astype(int)
    df[p+"stoch_B_bear"] = (cdn & mid).astype(int)
    return df

def add_mas(df, p=""):
    c = df["close"]
    mas = {
        "ema8":   c.ewm(span=EMA_FAST,  adjust=False).mean(),
        "ema21":  c.ewm(span=EMA_MID,   adjust=False).mean(),
        "sma50":  c.rolling(SMA_SLOW).mean(),
        "sma200": c.rolling(SMA_MACRO).mean(),
    }
    for name, ma in mas.items():
        dist  = (c - ma) / ma * 100
        abv   = (c > ma).astype(int)
        near  = (dist.abs() < 0.3).astype(int)
        cb    = ((abv==1)&(abv.shift(1)==0)).astype(int)
        cbear = ((abv==0)&(abv.shift(1)==1)).astype(int)
        rb    = ((df["low"]<=ma)&(c>ma)).astype(int)
        rbear = ((df["high"]>=ma)&(c<ma)).astype(int)
        std   = dist.rolling(200, min_periods=50).std()
        df[p+f"{name}"]              = ma
        df[p+f"dist_{name}_pct"]     = dist
        df[p+f"above_{name}"]        = abv
        df[p+f"near_{name}"]         = near
        df[p+f"cross_bull_{name}"]   = cb
        df[p+f"cross_bear_{name}"]   = cbear
        df[p+f"reject_bull_{name}"]  = rb
        df[p+f"reject_bear_{name}"]  = rbear
        df[p+f"bars_side_{name}"]    = bars_consecutive(abv==1) + bars_consecutive(abv==0)
        df[p+f"extreme_bull_{name}"] = (dist >  2*std).astype(int)
        df[p+f"extreme_bear_{name}"] = (dist < -2*std).astype(int)
    df[p+"align_full_bull"] = (
        (c>mas["ema8"])&(mas["ema8"]>mas["ema21"])&
        (mas["ema21"]>mas["sma50"])&(mas["sma50"]>mas["sma200"])
    ).astype(int)
    df[p+"align_full_bear"] = (
        (c<mas["ema8"])&(mas["ema8"]<mas["ema21"])&
        (mas["ema21"]<mas["sma50"])&(mas["sma50"]<mas["sma200"])
    ).astype(int)
    return df

def add_fib(df, p="", tf_name="1M"):
    n  = FIB_SWING_BARS.get(tf_name, 100)
    sh = df["high"].rolling(n, min_periods=5).max()
    sl = df["low"].rolling(n,  min_periods=5).min()
    sr = (sh - sl).replace(0, np.nan)
    c  = df["close"]
    emas = {
        "ema8":   c.ewm(span=EMA_FAST,  adjust=False).mean(),
        "ema21":  c.ewm(span=EMA_MID,   adjust=False).mean(),
        "sma50":  c.rolling(SMA_SLOW).mean(),
        "sma200": c.rolling(SMA_MACRO).mean(),
    }
    for ratio in FIB_LEVELS:
        lbl = str(ratio).replace(".", "")[:5]
        lvl = sl + sr * ratio
        dist = (c - lvl) / lvl * 100
        iz   = (dist.abs() < FIB_ZONE_PCT * 100).astype(int)
        rb   = ((df["low"]  <= lvl*(1+FIB_ZONE_PCT)) & (c > lvl)).astype(int)
        rbr  = ((df["high"] >= lvl*(1-FIB_ZONE_PCT)) & (c < lvl)).astype(int)
        df[p+f"fib{lbl}_level"]       = lvl
        df[p+f"fib{lbl}_dist_pct"]    = dist
        df[p+f"fib{lbl}_in_zone"]     = iz
        df[p+f"fib{lbl}_reject_bull"] = rb
        df[p+f"fib{lbl}_reject_bear"] = rbr
        df[p+f"fib{lbl}_above"]       = (c > lvl).astype(int)
        for mn, ma in emas.items():
            dma = (ma - lvl).abs() / lvl * 100
            df[p+f"fib{lbl}_conf_{mn}"] = (iz & (dma < 0.3)).astype(int)
    return df

def add_all_indicators(df, tf_name="1M"):
    df = df.copy()
    df = add_macd(df)
    df = add_stoch(df)
    df = add_mas(df)
    df = add_fib(df, tf_name=tf_name)
    return df
