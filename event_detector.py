import pandas as pd

EVENT_SUFFIXES = (
    "_macd_e1_bull","_macd_e1_bear","_macd_e2_bull","_macd_e2_bear",
    "_stoch_A_bull","_stoch_A_bear","_stoch_B_bull","_stoch_B_bear",
    "_cross_bull_ema21","_cross_bear_ema21","_cross_bull_sma200","_cross_bear_sma200",
    "_reject_bull_ema21","_reject_bear_ema21","_reject_bull_sma200","_reject_bear_sma200",
    "_extreme_bull_sma200","_extreme_bear_sma200",
    "_fib0382_in_zone","_fib05_in_zone","_fib0618_in_zone",
    "_fib0382_reject_bull","_fib05_reject_bull","_fib0618_reject_bull",
    "_fib0382_reject_bear","_fib05_reject_bear","_fib0618_reject_bear",
    "_fib05_conf_sma200","_fib0382_conf_ema21","_stoch_ob","_stoch_os",
)

def detect_events(state):
    event_cols = [
        c for c in state.columns
        if any(c.endswith(s) for s in EVENT_SUFFIXES)
        and state[c].dtype in ["int64","float64","int32"]
    ]
    if not event_cols:
        state["active_events"] = [[] for _ in range(len(state))]
        return state
    new_event = (state[event_cols].fillna(0).astype(int).diff() == 1)
    state["active_events"] = new_event.apply(lambda row: row.index[row].tolist(), axis=1)
    return state
