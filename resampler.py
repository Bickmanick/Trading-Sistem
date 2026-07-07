"""
Resampler — convierte el DataFrame 1m a los 8 TF del sistema.

Fix: eliminado el .shift(1).dropna() que desplazaba las barras
una posicion hacia el futuro (lookahead). El resample con
closed/label correctos ya garantiza que cada barra usa solo
la informacion disponible hasta su cierre.
"""
import pandas as pd
from config import TIMEFRAMES

OHLCV = {"open": "first", "high": "max", "low": "min",
          "close": "last", "volume": "sum"}


def resample_tf(df_1m: pd.DataFrame, tf_name: str) -> pd.DataFrame:
    rule = TIMEFRAMES[tf_name]["resample"]
    if tf_name == "1W":
        df = df_1m.resample(rule, closed="left", label="left").agg(OHLCV)
    else:
        df = df_1m.resample(rule, closed="right", label="right").agg(OHLCV)
    return df.dropna(subset=["close"])


def build_all_timeframes(df_1m: pd.DataFrame) -> dict:
    """
    Devuelve dict {tf_name: DataFrame OHLCV}.
    '1M' es una copia del 1m original.
    Los demas TF se generan por resample.
    """
    tfs = {"1M": df_1m.copy()}
    for tf_name in TIMEFRAMES:
        if tf_name != "1M":
            tfs[tf_name] = resample_tf(df_1m, tf_name)
    return tfs
