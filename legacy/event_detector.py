import pandas as pd

def detect_events(state: pd.DataFrame) -> pd.DataFrame:
    state = state.copy()
    state["event_bull"] = (
        state.get("1M_cross_bull_ema8", 0).astype(bool) |
        state.get("1M_cross_bull_ema21", 0).astype(bool)
    ).astype(int)
    state["event_bear"] = (
        state.get("1M_cross_bear_ema8", 0).astype(bool) |
        state.get("1M_cross_bear_ema21", 0).astype(bool)
    ).astype(int)
    return state
