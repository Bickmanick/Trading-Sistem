"""
MODULO 11 — GENERADOR DE INFORME

Genera dos archivos:
  1. data/results/{symbol}_report.html  — informe visual completo
  2. data/results/{symbol}_summary.csv  — tabla resumen de patrones

Contenido del informe HTML:
  - Resumen general: activo, periodo, N movimientos, N patrones
  - Tabla de patrones validados ordenados por WR validado
  - Distribucion de magnitudes y MFE/MAE por TF
  - Fibs de agotamiento y rebote mas frecuentes
  - Flag de robustez (de M10 si existe)
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from config import DATA_DIR

RESULTS_DIR = os.path.join(DATA_DIR, "results")


def _html_table(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty:
        return f"<h2>{title}</h2><p>Sin datos.</p>"
    rows = ""
    cols = list(df.columns)
    for _, r in df.iterrows():
        cells = "".join(f"<td>{r[c]}</td>" for c in cols)
        rows += f"<tr>{cells}</tr>\n"
    header = "".join(f"<th>{c}</th>" for c in cols)
    return f"""
<h2>{title}</h2>
<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse;font-size:12px;'>
  <thead><tr style='background:#1e3a5f;color:white'>{header}</tr></thead>
  <tbody>{rows}</tbody>
</table>
"""


def _summary_table(validated: pd.DataFrame) -> pd.DataFrame:
    keep = ["patron", "tf", "direccion", "N",
            "winrate", "wr_train", "wr_valid", "estado",
            "mag_media", "mfe_media", "mae_media",
            "rebote_fib_frecuente", "fib_agot_frecuente"]
    cols = [c for c in keep if c in validated.columns]
    df   = validated[cols].copy()
    if "wr_valid" in df.columns:
        df = df.sort_values("wr_valid", ascending=False, na_position="last")
    return df


def run_report(symbol: str,
               validated: pd.DataFrame,
               df_outcomes: pd.DataFrame) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    n_mov = len(df_outcomes) if df_outcomes is not None else 0
    n_pat = len(validated)   if validated   is not None else 0

    # ─── CSV resumen ───
    if validated is not None and not validated.empty:
        df_sum = _summary_table(validated)
        csv_path = os.path.join(RESULTS_DIR, f"{symbol}_summary.csv")
        df_sum.to_csv(csv_path, index=False)
        print(f"  M11: resumen CSV -> {csv_path}")
    else:
        df_sum = pd.DataFrame()

    # ─── Estadisticas por TF ───
    if df_outcomes is not None and not df_outcomes.empty:
        stats_tf = (df_outcomes.groupby(["tf", "direccion"])
                    .agg(
                        N=("mov_id", "count"),
                        mag_media=("magnitud_pct", "mean"),
                        mfe_media=("mfe_pct",      "mean"),
                        mae_media=("mae_pct",       "mean"),
                    )
                    .round(4)
                    .reset_index())
    else:
        stats_tf = pd.DataFrame()

    # ─── HTML ───
    html = f"""<!DOCTYPE html>
<html lang='es'>
<head>
  <meta charset='UTF-8'>
  <title>Informe {symbol} — Sistema Institucional</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
    h1   {{ color: #1e3a5f; }}
    h2   {{ color: #2c5f8a; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
    table {{ background: white; margin-bottom: 20px; }}
    tr:nth-child(even) {{ background: #f0f4f8; }}
    .badge-ok  {{ background: #27ae60; color: white; padding: 2px 6px; border-radius: 4px; }}
    .badge-pnd {{ background: #e67e22; color: white; padding: 2px 6px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Informe de Patrones — {symbol}</h1>
  <p>Generado: {now}</p>
  <ul>
    <li>Movimientos analizados: <strong>{n_mov}</strong></li>
    <li>Patrones validados: <strong>{n_pat}</strong></li>
  </ul>

  {_html_table(df_sum, 'Patrones Validados (ordenados por WR validado)')}
  {_html_table(stats_tf, 'Estadisticas por TF y Direccion')}

</body>
</html>"""

    html_path = os.path.join(RESULTS_DIR, f"{symbol}_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  M11: informe HTML -> {html_path}")
