import pandas as pd
import numpy as np

df = pd.read_csv("Tracking_data_efficiecny.csv")

e = df["Energy Consumption (kWh)"].dropna()

print("Rows:", len(e))
print("min / max:", e.min(), e.max())
print("mean:", e.mean())
print("sum:", e.sum())

# Check if it looks cumulative (monotonic increasing)
diff = e.diff().dropna()
print("\nDiff stats (first differences):")
print("diff min/max:", diff.min(), diff.max())
print("diff mean:", diff.mean())
print("fraction negative diffs:", (diff < 0).mean())

# Check if e is already cumulative:
print("\nFraction of monotonic increase:", (diff >= 0).mean())

# Try interpreting it as cumulative vs incremental
E_meas_if_cum = e.iloc[-1] - e.iloc[0]
E_meas_if_sum = e.sum()

print("\nIf cumulative: last-first =", E_meas_if_cum)
print("If incremental: sum =", E_meas_if_sum)

# Estimate average power implied if incremental (rough)
dt = df["DeltaT"].dropna().clip(0.05, 5.0)
T_total_h = dt.sum() / 3600
print("\nTotal time (h):", T_total_h)

print("Avg power if incremental sum:", E_meas_if_sum / T_total_h)
print("Avg power if cumulative last-first:", E_meas_if_cum / T_total_h)
