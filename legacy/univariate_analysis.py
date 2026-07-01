import pandas as pd
import numpy as np

def univariate_analysis(state: pd.DataFrame, min_edge: float = 0.02):
    bool_cols = [c for c in state.columns
                 if state[c].dtype in ("bool", "int64", "uint8", "float64")
                 and set(state[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
    records = []
    for col in bool_cols:
        sub = state[state[col] == 1]
        if len(sub) < 30:
            continue
        wr = (sub["close"].shift(-1) > sub["close"]).mean()
        edge = abs(wr - 0.5)
        if edge >= min_edge:
            records.append({"variable": col, "winrate": round(wr, 4), "N": len(sub), "edge": round(edge, 4)})
    df = pd.DataFrame(records).sort_values("edge", ascending=False)
    edge_vars = df[df["edge"] >= min_edge]["variable"].tolist()
    return edge_vars, df
