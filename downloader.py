import os, time, requests
import pandas as pd
from config import DATA_DIR, POLYGON_API_KEY

def csv_path(symbol):
    return os.path.join(DATA_DIR, f"{symbol}_1m.csv")

def csv_exists(symbol):
    p = csv_path(symbol)
    if not os.path.exists(p):
        return False
    try:
        df = pd.read_csv(p, nrows=1001)
        return len(df) > 1000
    except Exception:
        return False

def download(symbol, from_date, to_date):
    all_results = []
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute"
        f"/{from_date}/{to_date}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
    )
    while url:
        for retry in range(4):
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                break
            except Exception as e:
                time.sleep(2 ** retry)
        data = r.json()
        all_results.extend(data.get("results", []))
        url = data.get("next_url")
        if url:
            url += f"&apiKey={POLYGON_API_KEY}"
        time.sleep(0.25)
    df = pd.DataFrame(all_results)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    df = df[["timestamp","open","high","low","close","volume"]].set_index("timestamp").sort_index()
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(csv_path(symbol))
    return df

def ensure_data(symbol, from_date="2024-01-01", to_date="2026-06-17"):
    if csv_exists(symbol):
        return pd.read_csv(csv_path(symbol), index_col="timestamp", parse_dates=True)
    return download(symbol, from_date, to_date)
