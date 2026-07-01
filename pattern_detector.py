import numpy as np
import pandas as pd
from config import FORWARD_WINDOWS, KNN_THRESHOLD, KNN_MIN_OCC, KNN_AUTOCORR_H
from multiprocessing import freeze_support

def _process_chunk(chunk_indices, mat, close, state_index, state_meta, autocorr, knn_thr, knn_min, fwd_windows):
    records = []
    n_vars = mat.shape[1]
    n_close = len(close)
    for pos in chunk_indices:
        past_end = max(0, pos - autocorr)
        if past_end < knn_min:
            continue
        diffs  = (mat[:past_end] != mat[pos]).sum(axis=1)
        sims   = 1.0 - diffs / n_vars
        nbmask = sims >= knn_thr
        nn     = int(nbmask.sum())
        if nn < knn_min:
            continue
        nbpos = np.where(nbmask)[0]
        ts    = state_index[pos]
        row = {
            "ts":           ts,
            "n_neighbors":  nn,
            "regime":       state_meta["regime"][pos]       if "regime"       in state_meta else "",
            "thesis_long":  state_meta["thesis_long"][pos]  if "thesis_long"  in state_meta else "",
            "thesis_short": state_meta["thesis_short"][pos] if "thesis_short" in state_meta else "",
            "fib_conf":     int(state_meta["fib_conf_active"][pos])  if "fib_conf_active"  in state_meta else 0,
            "n_tfs_stoch_bear": int(state_meta["n_tfs_stoch_bear"][pos]) if "n_tfs_stoch_bear" in state_meta else 0,
            "n_tfs_stoch_bull": int(state_meta["n_tfs_stoch_bull"][pos]) if "n_tfs_stoch_bull" in state_meta else 0,
        }
        for w in fwd_windows:
            fp = nbpos + w
            valid = fp < n_close
            nb_v  = nbpos[valid]
            fp_v  = fp[valid]
            if len(nb_v) < knn_min:
                continue
            ep  = close[nb_v]
            fpp = close[fp_v]
            ret = (fpp - ep) / ep

            # MAE/MFE vectorizado: construir matriz de slices con stride tricks
            max_w   = w + 1
            nb_clip = nb_v[nb_v + max_w <= n_close]
            if len(nb_clip) > 0:
                idx_mat  = nb_clip[:, None] + np.arange(max_w)
                paths    = close[idx_mat]            # shape (n, w+1)
                ep_clip  = close[nb_clip][:, None]
                mae_vals = (paths.min(axis=1) - ep_clip.ravel()) / ep_clip.ravel() * 100
                mfe_vals = (paths.max(axis=1) - ep_clip.ravel()) / ep_clip.ravel() * 100
            else:
                mae_vals = mfe_vals = np.array([0.0])

            row[f"wr_long_{w}m"]  = round(float((ret > 0).mean()), 4)
            row[f"wr_short_{w}m"] = round(float((ret < 0).mean()), 4)
            row[f"pnl_long_{w}m"] = round(float(ret.mean() * 100), 4)
            row[f"mae_mean_{w}m"] = round(float(mae_vals.mean()), 4)
            row[f"mfe_mean_{w}m"] = round(float(mfe_vals.mean()), 4)
        records.append(row)
    return records


STATE_VARS = [
    "4H_macd_e1_bull","4H_macd_e1_bear","4H_macd_above_zero","4H_macd_above_sig",
    "1H_macd_e1_bull","1H_macd_e1_bear","1H_macd_above_zero",
    "30M_macd_e1_bull","30M_macd_e1_bear","1D_macd_above_zero","1D_macd_line_above_zero",
    "4H_stoch_A_bear","4H_stoch_ob","4H_stoch_A_bull","4H_stoch_os",
    "1H_stoch_A_bear","1H_stoch_A_bull","30M_stoch_B_bear","30M_stoch_B_bull",
    "15M_stoch_A_bull","15M_stoch_A_bear","5M_stoch_B_bull","5M_stoch_B_bear",
    "1M_stoch_A_bull","1M_stoch_A_bear",
    "4H_above_sma200","4H_near_sma200","4H_reject_bear_sma200",
    "4H_align_full_bull","4H_align_full_bear","1H_above_ema21","1H_reject_bull_ema21",
    "1D_above_sma200","1D_align_full_bear","1D_align_full_bull",
    "4H_fib05_in_zone","4H_fib0618_in_zone","4H_fib0382_in_zone",
    "4H_fib05_conf_sma200","4H_fib0382_conf_ema21","1H_fib0382_in_zone","1H_fib05_reject_bull",
    "casc_bear_macro","casc_bear_context","casc_bear_phase",
    "casc_bull_macro","casc_bull_context","casc_bull_phase","macro_indecision",
    "n_tfs_stoch_bear_ge2","n_tfs_stoch_bear_ge3","n_tfs_stoch_bull_ge2","n_tfs_stoch_bull_ge3",
]


def detect_patterns(state, edge_vars=None, n_jobs=8):
    from joblib import Parallel, delayed

    sv = [v for v in STATE_VARS if v in state.columns]
    if edge_vars:
        sv = [v for v in sv if v in edge_vars]
    if not sv:
        return pd.DataFrame()

    mat   = state[sv].fillna(0).astype(np.float32).values
    close = (state["close"] if "close" in state.columns
             else state[[c for c in state.columns if c.endswith("_close")][0]]).values

    autocorr = int(KNN_AUTOCORR_H * 60)

    tl  = state.get("thesis_long",  pd.Series("NONE", index=state.index))
    ts_ = state.get("thesis_short", pd.Series("NONE", index=state.index))
    cand_idx = np.where((tl.values != "NONE") | (ts_.values != "NONE"))[0]

    meta_cols = ["regime","thesis_long","thesis_short","fib_conf_active","n_tfs_stoch_bear","n_tfs_stoch_bull"]
    state_meta = {c: state[c].values for c in meta_cols if c in state.columns}
    state_index = state.index.values

    n_total = len(cand_idx)
    print(f"  Analizando {n_total:,} candidatos con {len(sv)} variables (n_jobs={n_jobs})...")

    chunks = np.array_split(cand_idx, max(1, n_jobs))

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=3)(
        delayed(_process_chunk)(
            chunk, mat, close, state_index, state_meta,
            autocorr, KNN_THRESHOLD, KNN_MIN_OCC, FORWARD_WINDOWS
        )
        for chunk in chunks if len(chunk) > 0
    )

    records = [r for res in results for r in res]
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).set_index("ts")
    print(f"  {len(df):,} patrones encontrados")
    return df
