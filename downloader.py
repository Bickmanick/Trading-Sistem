import os, time, requests
from datetime import datetime, timezone
import pandas as pd
from config import DATA_DIR, POLYGON_API_KEY


def csv_path(symbol):
    return os.path.join(DATA_DIR, f"{symbol}_1m.csv")


def _read_existing(symbol):
    """Lee el CSV local si existe y devuelve el DataFrame o None."""
    p = csv_path(symbol)
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_csv(p, index_col="timestamp", parse_dates=True).sort_index()
        return df if not df.empty else None
    except Exception:
        return None


def download(symbol, from_date, to_date):
    """Descarga solo el rango que falta y lo une al histórico existente."""
    existing = _read_existing(symbol)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    to_date = to_date or today

    start = pd.to_datetime(from_date, utc=True)
    end   = pd.to_datetime(to_date,   utc=True)

    # Si ya tenemos datos, solo descargamos desde la última fecha
    if existing is not None:
        last = existing.index.max()
        if last >= end:
            print(f"  CSV ya actualizado hasta {last.date()}. Sin descarga.")
            return existing
        start = last + pd.Timedelta(minutes=1)
        print(f"  Actualizando desde {start.date()} hasta {end.date()}...")
    else:
        print(f"  Descarga completa: {start.date()} → {end.date()}")

    all_results = []
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute"
        f"/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
    )

    while url:
        data = None
        last_exc = None
        for retry in range(4):
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                last_exc = e
                time.sleep(2 ** retry)
        if data is None:
            # Si falla la API pero tenemos datos locales, los devolvemos
            if existing is not None:
                print(f"  AVISO: API no disponible. Usando histórico local.")
                return existing
            raise RuntimeError(f"Polygon no responde y no hay CSV local: {last_exc}")

        all_results.extend(data.get("results", []))
        url = data.get("next_url")
        if url:
            url += f"&apiKey={POLYGON_API_KEY}"
        time.sleep(0.25)

    # Si la API no devuelve barras nuevas pero ya tenemos histórico
    if not all_results:
        if existing is not None:
            print("  AVISO: Sin datos nuevos en la API. Usando histórico local.")
            return existing
        raise RuntimeError("Sin datos en Polygon y sin CSV local. Verifica API key y ticker.")

    # Construir DataFrame con los datos nuevos
    df_new = pd.DataFrame(all_results)
    df_new["timestamp"] = pd.to_datetime(df_new["t"], unit="ms", utc=True)
    df_new = df_new.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df_new = df_new[["timestamp", "open", "high", "low", "close", "volume"]].set_index("timestamp").sort_index()

    # Combinar con histórico existente
    if existing is not None:
        df_new = pd.concat([existing, df_new])
        df_new = df_new[~df_new.index.duplicated(keep="last")].sort_index()

    os.makedirs(DATA_DIR, exist_ok=True)
    df_new.to_csv(csv_path(symbol))
    print(f"  {len(df_new):,} velas totales guardadas en {csv_path(symbol)}")
    return df_new


def ensure_data(symbol, from_date="2024-01-01", to_date=None):
    """Punto de entrada principal. Siempre intenta actualizar hasta hoy."""
    to_date = to_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return download(symbol, from_date, to_date)
