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


def _polygon_fetch(symbol, start: pd.Timestamp, end: pd.Timestamp, existing):
    """Descarga de Polygon el rango [start, end] y lo une al historico."""
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
            if existing is not None:
                print(f"  AVISO: API no disponible. Usando historico local ({len(existing):,} velas).")
                return existing
            raise RuntimeError(f"Polygon no responde y no hay CSV local: {last_exc}")

        all_results.extend(data.get("results", []))
        url = data.get("next_url")
        if url:
            url += f"&apiKey={POLYGON_API_KEY}"
        time.sleep(0.25)

    if not all_results:
        if existing is not None:
            print("  Sin datos nuevos en API. Usando historico local.")
            return existing
        raise RuntimeError("Sin datos en Polygon y sin CSV local. Verifica API key y ticker.")

    df_new = pd.DataFrame(all_results)
    df_new["timestamp"] = pd.to_datetime(df_new["t"], unit="ms", utc=True)
    df_new = df_new.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df_new = df_new[["timestamp", "open", "high", "low", "close", "volume"]].set_index("timestamp").sort_index()

    if existing is not None:
        df_new = pd.concat([existing, df_new])
        df_new = df_new[~df_new.index.duplicated(keep="last")].sort_index()

    os.makedirs(DATA_DIR, exist_ok=True)
    df_new.to_csv(csv_path(symbol))
    print(f"  {len(df_new):,} velas totales guardadas.")
    return df_new


def download(symbol, from_date, to_date):
    """Descarga inteligente: solo el rango que realmente falta."""
    existing = _read_existing(symbol)
    today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    to_date  = to_date or today

    start_req = pd.to_datetime(from_date, utc=True)
    end_req   = pd.to_datetime(to_date,   utc=True)

    if existing is not None:
        first_local = existing.index.min()
        last_local  = existing.index.max()

        # Necesitamos datos ANTES de lo que tenemos (historico mas largo)
        need_past   = start_req < (first_local - pd.Timedelta(days=1))
        # Necesitamos datos DESPUES de lo que tenemos (actualizacion)
        need_future = last_local < (end_req   - pd.Timedelta(hours=1))

        if not need_past and not need_future:
            print(f"  CSV al dia: {first_local.date()} -> {last_local.date()} ({len(existing):,} velas). Sin descarga.")
            # Recortar al rango pedido
            return existing[(existing.index >= start_req) & (existing.index <= end_req)]

        if need_past and need_future:
            # Descargar todo el rango completo
            print(f"  Descarga completa: {start_req.date()} -> {end_req.date()}")
            return _polygon_fetch(symbol, start_req, end_req, existing)

        if need_past:
            # Descargar desde start_req hasta el primer dato local
            fetch_end = first_local - pd.Timedelta(minutes=1)
            print(f"  Descargando historico anterior: {start_req.date()} -> {fetch_end.date()}")
            return _polygon_fetch(symbol, start_req, fetch_end, existing)

        if need_future:
            # Solo actualizar desde la ultima barra
            fetch_start = last_local + pd.Timedelta(minutes=1)
            print(f"  Actualizando: {fetch_start.date()} -> {end_req.date()}")
            return _polygon_fetch(symbol, fetch_start, end_req, existing)

    else:
        print(f"  Descarga completa: {start_req.date()} -> {end_req.date()}")
        return _polygon_fetch(symbol, start_req, end_req, None)


def ensure_data(symbol, from_date="2024-01-01", to_date=None):
    """Punto de entrada principal. Garantiza que el CSV cubre [from_date, to_date]."""
    to_date = to_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    df = download(symbol, from_date, to_date)
    # Devolver solo el rango pedido
    start = pd.to_datetime(from_date, utc=True)
    end   = pd.to_datetime(to_date,   utc=True)
    return df[(df.index >= start) & (df.index <= end)]
