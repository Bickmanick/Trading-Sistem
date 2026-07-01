import pandas as pd

def build_state_matrix(df_1m, tfs):
    state = df_1m[["open","high","low","close"]].copy()
    for tf_name, df_tf in tfs.items():
        if tf_name == "1M":
            ind_cols = [c for c in df_tf.columns if c not in ["open","high","low","close","volume"]]
            for col in ind_cols:
                state[f"1M_{col}"] = df_tf[col].reindex(state.index, method="ffill")
            continue
        df_tf_pref = df_tf.add_prefix(f"{tf_name}_").sort_index()
        state = pd.merge_asof(
            state.sort_index(), df_tf_pref,
            left_index=True, right_index=True, direction="backward",
        )
    return state
