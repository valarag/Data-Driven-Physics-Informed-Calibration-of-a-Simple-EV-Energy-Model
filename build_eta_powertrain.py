import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_TRACKING_FILE = "Tracking_data_efficiecny.csv"
OUTPUT_DIR = Path("curves")
OUTPUT_DIR.mkdir(exist_ok=True)

SPEED_BIN_WIDTH = 5  # km/h
MIN_SAMPLES_PER_BIN = 50  # discard bins with too few samples

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(INPUT_TRACKING_FILE)

# Keep only necessary columns
df = df[["Speed", "Powertrain_efficiency_gear_SG"]].copy()
df.columns = ["speed_kph", "eta_powertrain"]

# ==========================================================
# CLEANING
# ==========================================================

# Remove NaNs
df = df.dropna()

# Remove non-physical values
df = df[df["eta_powertrain"] > 0]
df = df[df["eta_powertrain"] < 1.05]

# Remove unrealistic high efficiency (sensor artifacts)
df = df[df["eta_powertrain"] < 0.99]

# Remove negative or extreme speeds
df = df[(df["speed_kph"] >= 0) & (df["speed_kph"] < 200)]

print(f"Remaining samples: {len(df)}")

# ==========================================================
# BINNING BY SPEED
# ==========================================================

bins = np.arange(0, 160 + SPEED_BIN_WIDTH, SPEED_BIN_WIDTH)

df["speed_bin"] = pd.cut(df["speed_kph"], bins)

eta_curve = df.groupby("speed_bin")["eta_powertrain"].agg(
    eta_powertrain_mean="mean",
    eta_powertrain_p05=lambda x: np.percentile(x, 5),
    eta_powertrain_p95=lambda x: np.percentile(x, 95),
    n_samples="count"
).reset_index()

# Remove bins with insufficient samples
eta_curve = eta_curve[eta_curve["n_samples"] >= MIN_SAMPLES_PER_BIN]

# Convert bins to midpoint speed
eta_curve["speed_kph"] = eta_curve["speed_bin"].apply(
    lambda x: x.mid if pd.notnull(x) else np.nan
)

eta_curve = eta_curve.drop(columns=["speed_bin"])

# Clip physically realistic bounds
eta_curve["eta_powertrain_mean"] = eta_curve["eta_powertrain_mean"].clip(0.75, 0.98)

# ==========================================================
# OPTIONAL: LOWESS SMOOTHING
# ==========================================================
"""
WHY SMOOTH?

Raw binned mean can fluctuate due to:
- Uneven sampling
- Driving cycle structure
- Noise

LOWESS smoothing:
- Preserves general shape
- Reduces oscillations
- Keeps physical realism

This is OPTIONAL but recommended for calibration curves.
"""

APPLY_SMOOTHING = True

if APPLY_SMOOTHING:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    smoothed = lowess(
        eta_curve["eta_powertrain_mean"],
        eta_curve["speed_kph"],
        frac=0.25  # smoothing strength (0.2–0.3 recommended)
    )
    
    eta_curve["eta_powertrain_mean"] = smoothed[:, 1]

# ==========================================================
# SAVE
# ==========================================================

output_file = OUTPUT_DIR / "eta_powertrain_vs_speed.csv"
eta_curve.to_csv(output_file, index=False)

print(f"Saved: {output_file}")

import matplotlib.pyplot as plt

plt.figure()
plt.plot(eta_curve["speed_kph"], eta_curve["eta_powertrain_mean"])
plt.fill_between(
    eta_curve["speed_kph"],
    eta_curve["eta_powertrain_p05"],
    eta_curve["eta_powertrain_p95"],
    alpha=0.2
)
plt.xlabel("Speed (km/h)")
plt.ylabel("Powertrain efficiency")
plt.title("η_powertrain(v) calibration curve")
plt.show()
