import pandas as pd

TFS = ["1W", "1D", "4H", "1H", "30M", "15M", "5M"]

def compute_alignment(state: pd.DataFrame) -> pd.DataFrame:
    state = state.copy()
    bull_score = sum(
        state.get(f"{tf}_above_ema21", pd.Series(0, index=state.index)).fillna(0)
        for tf in TFS
    )
    bear_score = sum(
        (1 - state.get(f"{tf}_above_ema21", pd.Series(1, index=state.index)).fillna(1))
        for tf in TFS
    )
    state["bull_alignment"] = bull_score
    state["bear_alignment"] = bear_score
    state["regime"] = "NEUTRO"
    state.loc[bull_score >= 5, "regime"] = "BULL"
    state.loc[bear_score >= 5, "regime"] = "BEAR"
    return state
