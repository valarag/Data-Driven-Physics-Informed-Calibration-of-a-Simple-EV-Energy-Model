import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

matplotlib_mode = True

# =========================
# CONFIG
# =========================
ROOT = Path("2_preprocessed")
OUTDIR = Path("curves")
OUTDIR.mkdir(exist_ok=True)

TEMP_BIN_C = 2.0                 # temperature bin width (°C)
WINDOW_S = 1.0                   # window duration for dV/dI (seconds)
FS_HZ = 10                       # dataset is strict 10 Hz
WIN = int(WINDOW_S * FS_HZ)      # samples per window

MIN_SAMPLES_PER_BIN = 5000       # robustness
I_MIN_A = 0.5                    # ignore near-zero currents (noise dominates)
ETA_CLIP = (0.70, 0.995)         # physical bounds for proxy


# =========================
# HELPERS
# =========================
def robust_dv_di(df):
    """
    Estimate R ~ dV/dI on rolling windows.
    Uses difference between endpoints in the window.
    """
    V = df["Voltage"].to_numpy()
    I = df["Current"].to_numpy()

    # Rolling endpoint differences
    dV = V[WIN:] - V[:-WIN]
    dI = I[WIN:] - I[:-WIN]

    # Avoid divide by zero
    mask = np.abs(dI) > 1e-3
    R = np.full_like(dV, np.nan, dtype=float)
    R[mask] = dV[mask] / dI[mask]
    return R


# =========================
# LOAD ALL FILES (preprocessed only)
# =========================
files = sorted(ROOT.rglob("*.csv"))
if len(files) == 0:
    raise FileNotFoundError(f"No CSV found under {ROOT.resolve()}")

rows = []
for f in files:
    df = pd.read_csv(f, usecols=["Time", "Voltage", "Current", "Temperature", "SOC"])

    # IMPORTANT: discharge current is negative in your dataset
    df = df[df["Current"] < -I_MIN_A].copy()

    if len(df) < (WIN + 5):
        continue

    # Estimate R(t) from dV/dI
    R = robust_dv_di(df)
    df_mid = df.iloc[WIN:].copy()
    df_mid["R_ohm_est"] = R

    # Compute output power during discharge (positive magnitude)
    P_out_W = -(df_mid["Voltage"] * df_mid["Current"])  # Current negative => -V*I > 0
    P_loss_W = (df_mid["Current"] ** 2) * df_mid["R_ohm_est"]

    # Battery efficiency proxy
    eta = P_out_W / (P_out_W + P_loss_W)

    df_mid["eta_battery"] = eta

    # Keep only valid
    df_mid = df_mid.replace([np.inf, -np.inf], np.nan).dropna(subset=["eta_battery"])
    df_mid = df_mid[(df_mid["eta_battery"] >= ETA_CLIP[0]) & (df_mid["eta_battery"] <= ETA_CLIP[1])]

    # Store minimal columns
    rows.append(df_mid[["Temperature", "eta_battery"]])

print(f"Loaded files: {len(files)}")
df_all = pd.concat(rows, ignore_index=True)
print("Samples after filtering:", len(df_all))

# =========================
# BIN BY TEMPERATURE
# =========================
tmin = np.floor(df_all["Temperature"].min())
tmax = np.ceil(df_all["Temperature"].max())
bins = np.arange(tmin, tmax + TEMP_BIN_C, TEMP_BIN_C)

df_all["T_bin"] = pd.cut(df_all["Temperature"], bins)

curve = df_all.groupby("T_bin")["eta_battery"].agg(
    eta_battery_mean="mean",
    eta_battery_p05=lambda x: np.percentile(x, 5),
    eta_battery_p95=lambda x: np.percentile(x, 95),
    n_samples="count"
).reset_index()

curve = curve[curve["n_samples"] >= MIN_SAMPLES_PER_BIN].copy()

curve["T_bat_C"] = curve["T_bin"].apply(lambda x: x.mid if pd.notnull(x) else np.nan)
curve = curve.drop(columns=["T_bin"])

# =========================
# OPTIONAL: SMOOTHING (keep OFF by default)
# =========================
APPLY_SMOOTHING = False
if APPLY_SMOOTHING and len(curve) >= 5:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    sm = lowess(curve["eta_battery_mean"], curve["T_bat_C"], frac=0.35)
    curve["eta_battery_mean"] = sm[:, 1]

# =========================
# SAVE
# =========================
out = OUTDIR / "eta_battery_vs_Tbat.csv"
curve.to_csv(out, index=False)
print("Saved:", out)

# =========================
# QUICK PLOT
# =========================
if matplotlib_mode:
    plt.figure()
    plt.plot(curve["T_bat_C"], curve["eta_battery_mean"])
    plt.fill_between(curve["T_bat_C"], curve["eta_battery_p05"], curve["eta_battery_p95"], alpha=0.2)
    plt.xlabel("Battery temperature (°C)")
    plt.ylabel("Battery efficiency (proxy)")
    plt.title("η_battery(T_bat) calibration curve")
    plt.show()

else:
    # Base line (mean curve)
    fig = px.line(
        curve,
        x="T_bat_C",
        y="eta_battery_mean",
        labels={
            "T_bat_C": "Battery temperature (°C)",
            "eta_battery_mean": "Battery efficiency (proxy)"
        },
        title="η_battery(T_bat) calibration curve"
    )

    # Add shaded confidence band (p05–p95)
    fig.add_trace(
        go.Scatter(
            x=curve["T_bat_C"],
            y=curve["eta_battery_p95"],
            mode="lines",
            line=dict(width=0),
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=curve["T_bat_C"],
            y=curve["eta_battery_p05"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line=dict(width=0),
            name="p05–p95 range"
        )
    )

    fig.show()