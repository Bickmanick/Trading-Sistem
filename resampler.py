import pandas as pd
from config import TIMEFRAMES

OHLCV = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}

def resample_tf(df_1m, tf_name):
    rule = TIMEFRAMES[tf_name]["resample"]
    if tf_name == "1W":
        df = df_1m.resample(rule, closed="left", label="left").agg(OHLCV)
    else:
        df = df_1m.resample(rule, closed="right", label="right").agg(OHLCV)
    df = df.dropna(subset=["close"]).shift(1).dropna(subset=["close"])
    return df

def build_all_timeframes(df_1m):
    tfs = {"1M": df_1m.copy()}
    for tf_name in list(TIMEFRAMES.keys())[:-1]:
        tfs[tf_name] = resample_tf(df_1m, tf_name)
    return tfs
