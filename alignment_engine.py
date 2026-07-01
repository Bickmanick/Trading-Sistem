import numpy as np
import pandas as pd
from config import TF_NAMES, MACRO_TFS, CONTEXT_TFS, PHASE_TFS, EXEC_TFS

def _safe(s): return s.fillna(0).astype(int)

def compute_alignment(state):
    def col(tf, name):
        c = f"{tf}_{name}"
        return _safe(state[c]) if c in state.columns else pd.Series(0, index=state.index)

    biases = {}
    for tf in TF_NAMES:
        bull = ((col(tf,"above_ema21")==1)&(col(tf,"above_sma50")==1)&
                (col(tf,"macd_line_above_zero")==1)&(col(tf,"stoch_K")>50))
        bear = ((col(tf,"above_ema21")==0)&(col(tf,"above_sma50")==0)&
                (col(tf,"macd_line_above_zero")==0)&(col(tf,"stoch_K")<50))
        biases[tf] = pd.Series("NEUTRAL", index=state.index)
        biases[tf] = biases[tf].where(~bull,"BULL")
        biases[tf] = biases[tf].where(~bear,"BEAR")
        state[f"bias_{tf}"] = biases[tf]

    def cb(tfs): return np.logical_and.reduce([(biases[t]=="BULL").values for t in tfs])
    def cbr(tfs): return np.logical_and.reduce([(biases[t]=="BEAR").values for t in tfs])

    state["casc_bull_macro"]   = cb(MACRO_TFS).astype(int)
    state["casc_bear_macro"]   = cbr(MACRO_TFS).astype(int)
    state["casc_bull_context"] = cb(CONTEXT_TFS).astype(int)
    state["casc_bear_context"] = cbr(CONTEXT_TFS).astype(int)
    state["casc_bull_phase"]   = cb(PHASE_TFS).astype(int)
    state["casc_bear_phase"]   = cbr(PHASE_TFS).astype(int)
    state["casc_bull_exec"]    = cb(EXEC_TFS).astype(int)
    state["casc_bear_exec"]    = cbr(EXEC_TFS).astype(int)
    state["casc_bull_full"]    = ((state["casc_bull_macro"]==1)&(state["casc_bull_context"]==1)&
                                   (state["casc_bull_phase"]==1)&(state["casc_bull_exec"]==1)).astype(int)
    state["casc_bear_full"]    = ((state["casc_bear_macro"]==1)&(state["casc_bear_context"]==1)&
                                   (state["casc_bear_phase"]==1)&(state["casc_bear_exec"]==1)).astype(int)
    state["macro_indecision"]  = ((state["casc_bull_macro"]==0)&(state["casc_bear_macro"]==0)).astype(int)

    for dir_ in ["bull","bear"]:
        sb_cols  = [f"{tf}_stoch_A_{dir_}" for tf in TF_NAMES if f"{tf}_stoch_A_{dir_}" in state.columns]
        sb_cols += [f"{tf}_stoch_B_{dir_}" for tf in TF_NAMES if f"{tf}_stoch_B_{dir_}" in state.columns]
        if sb_cols:
            n = state[sb_cols].fillna(0).astype(int).sum(axis=1)
            state[f"n_tfs_stoch_{dir_}"]     = n
            state[f"n_tfs_stoch_{dir_}_ge2"] = (n>=2).astype(int)
            state[f"n_tfs_stoch_{dir_}_ge3"] = (n>=3).astype(int)

    regime = pd.Series("RANGE", index=state.index)
    regime = regime.where(state["casc_bull_full"]!=1,"IMPULSE_BULL")
    regime = regime.where(state["casc_bear_full"]!=1,"IMPULSE_BEAR")
    eb = ((state["casc_bull_macro"]==1)&(state["casc_bull_context"]==1)&
          ((state["casc_bear_phase"]==1)|(state["casc_bear_exec"]==1)))
    ebr= ((state["casc_bear_macro"]==1)&(state["casc_bear_context"]==1)&
          ((state["casc_bull_phase"]==1)|(state["casc_bull_exec"]==1)))
    regime = regime.where(~eb,"EXHAUST_BULL")
    regime = regime.where(~ebr,"EXHAUST_BEAR")
    fe = ((state["casc_bull_macro"]==1)&(state["casc_bull_phase"]==1)&(state["casc_bear_exec"]==1))
    fer= ((state["casc_bear_macro"]==1)&(state["casc_bear_phase"]==1)&(state["casc_bull_exec"]==1))
    regime = regime.where(~fe,"FALSE_EXHAUST_BULL")
    regime = regime.where(~fer,"FALSE_EXHAUST_BEAR")
    conf = (((state["casc_bull_macro"]==1)&(state["casc_bear_exec"]==1))|
            ((state["casc_bear_macro"]==1)&(state["casc_bull_exec"]==1)))
    regime = regime.where(~conf,"CONFLICT")
    ind = ((state["macro_indecision"]==1)&
           ((state["casc_bull_context"]==1)|(state["casc_bear_context"]==1)))
    regime = regime.where(~ind,"INDECISION_MACRO")
    state["regime"] = regime
    return state
