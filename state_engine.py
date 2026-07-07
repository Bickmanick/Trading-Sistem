"""
State Engine — construye el state matrix completo.

Fix: todos los TF reciben prefijo <TF>_ incluyendo 1M,
y se fuerza timezone UTC antes del merge_asof para evitar
errores con indices TZ-naive vs TZ-aware.
"""
import pandas as pd


def build_state_matrix(df_1m: pd.DataFrame, tfs: dict) -> pd.DataFrame:
    """
    Construye una tabla indexada en 1M donde cada columna es
    <TF>_<indicador>, rellenando hacia adelante (ffill) el
    valor del TF superior sobre cada barra de 1 minuto.

    Parameters
    ----------
    df_1m : DataFrame OHLCV base (index = timestamp UTC)
    tfs   : dict {tf_name: DataFrame con indicadores calculados}

    Returns
    -------
    DataFrame indexado en 1M con todas las columnas de estado.
    """
    # Base: OHLC del 1m
    state = df_1m[["open", "high", "low", "close"]].copy()

    # Asegurar TZ UTC en el indice base
    if state.index.tz is None:
        state.index = state.index.tz_localize("UTC")
    state = state.sort_index()

    for tf_name, df_tf in tfs.items():
        # Columnas de indicadores (excluir OHLCV)
        ind_cols = [c for c in df_tf.columns
                    if c not in ("open", "high", "low", "close", "volume")]
        if not ind_cols:
            continue

        df_pref = df_tf[ind_cols].copy()
        df_pref.columns = [f"{tf_name}_{c}" for c in ind_cols]

        # Asegurar TZ UTC
        if df_pref.index.tz is None:
            df_pref.index = df_pref.index.tz_localize("UTC")
        df_pref = df_pref.sort_index()
        df_pref.index.name = "timestamp"

        if tf_name == "1M":
            # Reindexar directamente (mismo indice)
            for col in df_pref.columns:
                state[col] = df_pref[col].reindex(state.index, method="ffill")
        else:
            # merge_asof: para cada barra 1M, toma el ultimo valor del TF
            state_reset = state.reset_index()
            pref_reset  = df_pref.reset_index()
            merged = pd.merge_asof(
                state_reset.sort_values("timestamp"),
                pref_reset.sort_values("timestamp"),
                on="timestamp",
                direction="backward"
            )
            merged = merged.set_index("timestamp")
            for col in df_pref.columns:
                if col in merged.columns:
                    state[col] = merged[col].values

    return state
