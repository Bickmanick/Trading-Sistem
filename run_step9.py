import pandas as pd
from pathlib import Path
from trade_cycle import run_trade_cycle

symbol     = "NVDA"
output_dir = Path("output") / symbol

print("Cargando archivos...")
state_matrix = pd.read_csv(output_dir / "state_matrix.csv", index_col=0)
patterns     = pd.read_csv(output_dir / "patterns.csv")
univariate   = pd.read_csv(output_dir / "univariate.csv")
print(f"  state_matrix: {state_matrix.shape}")
print(f"  patterns:     {len(patterns):,} filas")
print(f"  univariate:   {len(univariate):,} filas")

run_trade_cycle(state_matrix, patterns, univariate, symbol, output_dir)
