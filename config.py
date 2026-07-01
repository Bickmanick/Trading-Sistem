import os
import warnings
warnings.filterwarnings("ignore")

TIMEFRAMES = {
    "1W":  {"role": "macro",   "resample": "W-MON", "minutes": 10080},
    "1D":  {"role": "macro",   "resample": "1D",    "minutes": 1440},
    "4H":  {"role": "context", "resample": "4h",    "minutes": 240},
    "1H":  {"role": "context", "resample": "1h",    "minutes": 60},
    "30M": {"role": "phase",   "resample": "30min", "minutes": 30},
    "15M": {"role": "phase",   "resample": "15min", "minutes": 15},
    "5M":  {"role": "exec",    "resample": "5min",  "minutes": 5},
    "1M":  {"role": "exec",    "resample": "1min",  "minutes": 1},
}
TF_NAMES = list(TIMEFRAMES.keys())
MACRO_TFS   = ["1W","1D"]
CONTEXT_TFS = ["4H","1H"]
PHASE_TFS   = ["30M","15M"]
EXEC_TFS    = ["5M","1M"]

MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
STOCH_K, STOCH_D, STOCH_SMOOTH    = 14, 3, 3
STOCH_OB, STOCH_OS                = 80, 20
EMA_FAST, EMA_MID                 = 8, 21
SMA_SLOW, SMA_MACRO               = 50, 200

FIB_LEVELS   = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
FIB_ZONE_PCT = 0.003
FIB_SWING_BARS = {
    "1W": 20, "1D": 20, "4H": 50, "1H": 50,
    "30M": 100, "15M": 100, "5M": 200, "1M": 300,
}

KNN_THRESHOLD  = 0.75
KNN_MIN_OCC    = 10
KNN_AUTOCORR_H = 2

FORWARD_WINDOWS = [30, 60, 120]
EDGE_WINRATE    = 0.55
EDGE_MIN_OCC    = 10

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
POLYGON_API_KEY = 'mLYwdfVUko1UuFnTInckjMRiYodIFx0u'
