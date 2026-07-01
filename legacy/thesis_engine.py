import pandas as pd

def compute_thesis(state: pd.DataFrame) -> pd.DataFrame:
    state = state.copy()
    state["thesis_long"]  = (state.get("bull_alignment", 0) >= 4).astype(int)
    state["thesis_short"] = (state.get("bear_alignment", 0) >= 4).astype(int)
    return state
