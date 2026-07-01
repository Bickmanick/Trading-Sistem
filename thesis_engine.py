import pandas as pd

VALID_LONG  = {"IMPULSE_BULL","EXHAUST_BEAR","FALSE_EXHAUST_BEAR","INDECISION_MACRO"}
VALID_SHORT = {"IMPULSE_BEAR","EXHAUST_BULL","FALSE_EXHAUST_BULL","INDECISION_MACRO"}

def _c(state, name):
    return state[name] if name in state.columns else pd.Series(0, index=state.index)

def compute_thesis(state):
    fib_conf = (
        (_c(state,"4H_fib05_conf_sma200")==1)|(_c(state,"4H_fib0382_conf_ema21")==1)|
        (_c(state,"4H_fib0618_conf_sma50")==1)|(_c(state,"1H_fib0382_in_zone")==1)|
        (_c(state,"1H_fib05_in_zone")==1)
    )
    mb  = _c(state,"casc_bull_macro")==1
    mr  = _c(state,"casc_bear_macro")==1
    cb  = _c(state,"casc_bull_context")==1
    cr  = _c(state,"casc_bear_context")==1
    sb2 = _c(state,"n_tfs_stoch_bull_ge2")==1
    sr2 = _c(state,"n_tfs_stoch_bear_ge2")==1
    mi  = _c(state,"macro_indecision")==1
    reg = state["regime"] if "regime" in state.columns else pd.Series("RANGE",index=state.index)

    th_long = pd.Series("NONE",index=state.index)
    th_long = th_long.where(~(mi&cb&(reg=="INDECISION_MACRO")), "BAJA")
    th_long = th_long.where(~(mb&cb&reg.isin(VALID_LONG)),       "MEDIA")
    th_long = th_long.where(~(mb&cb&sb2&fib_conf&reg.isin(VALID_LONG)),"ALTA")

    th_short = pd.Series("NONE",index=state.index)
    th_short = th_short.where(~(mi&cr&(reg=="INDECISION_MACRO")), "BAJA")
    th_short = th_short.where(~(mr&cr&reg.isin(VALID_SHORT)),      "MEDIA")
    th_short = th_short.where(~(mr&cr&sr2&fib_conf&reg.isin(VALID_SHORT)),"ALTA")

    state["thesis_long"]     = th_long
    state["thesis_short"]    = th_short
    state["fib_conf_active"] = fib_conf.astype(int)
    return state
