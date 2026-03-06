"""
build_eta_powertrain.py
=======================
Builds the powertrain efficiency curve  eta_powertrain(speed)
from the primary OBD trip recording.

WHAT CHANGED FROM PREVIOUS VERSION
------------------------------------
The previous version used the Powertrain_efficiency_gear_SG column directly.
That column captures ONLY motor/gear mechanical efficiency (~85%).
It misses battery internal resistance, power electronics, DC-DC converters,
thermal management, and all other auxiliaries — which together account for
the gap between mechanical energy and actual battery energy drawn.

This version computes TRUE SYSTEM EFFICIENCY empirically:

    eta_sys(v) = P_mech(v) / P_battery(v)

Where:
    P_mech(v)    = (F_aero + F_rr + F_grade + F_inertia) × v   [parametric]
    P_battery(v) = Energy Consumption (kWh) / dt                [measured]

This is the only efficiency definition that makes the physics model
predict the correct battery energy draw.

The curve is built by binning rows by speed and computing the ratio
in each bin — giving a speed-dependent eta_sys curve that the
physics model can interpolate at runtime.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_FILE      = "Tracking_data_efficiecny.csv"
OUTPUT_DIR      = Path("curves")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE     = OUTPUT_DIR / "eta_powertrain_vs_speed.csv"

SPEED_BIN_WIDTH = 5        # km/h per bin
MIN_SAMPLES     = 100      # minimum rows per bin to be included

# Fixed vehicle parameters (Tesla Model 3 LR AWD)
RHO  = 1.225   # kg/m³
CD   = 0.23    # drag coefficient  (initial estimate — calibrate_physics_model
               # will tune this, but we need a starting point for the curve)
A    = 2.22    # m²
CRR  = 0.009   # rolling resistance
MASS = 1847    # kg
G    = 9.81    # m/s²

# Power clip — remove physically impossible rows before binning
# Tesla Model 3 peak battery power ~250 kW, typical driving 5-50 kW
P_BAT_MAX_W  =  250_000   #  250 kW discharge
P_BAT_MIN_W  = -100_000   # -100 kW regen
P_MECH_MIN_W = -999_999   # include all rows, not just positive work
                           # (avoids division issues in regen / standstill)

# ==========================================================
# LOAD & CLEAN
# ==========================================================

print(f"Loading {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE)

required = ["Speed", "DeltaT", "Energy Consumption (kWh)",
            "Slope Angle (rad)", "Acceleration"]
df = df.dropna(subset=required).copy()

df["dt_s"]   = pd.to_numeric(df["DeltaT"],              errors="coerce").clip(0.05, 5.0)
df["v_mps"]  = pd.to_numeric(df["Speed"],               errors="coerce").clip(lower=0) / 3.6
df["theta"]  = pd.to_numeric(df["Slope Angle (rad)"],   errors="coerce").fillna(0)
df["accel"]  = pd.to_numeric(df["Acceleration"],        errors="coerce").fillna(0)
df["energy"] = pd.to_numeric(df["Energy Consumption (kWh)"], errors="coerce")

# Drop the one NaN dt_s row
df = df.dropna(subset=["dt_s", "v_mps", "energy"])

print(f"  Rows after cleaning: {len(df):,}")

# ==========================================================
# COMPUTE MECHANICAL POWER (parametric)
# ==========================================================

v     = df["v_mps"].values
theta = df["theta"].values
a     = df["accel"].values

F_aero = 0.5 * RHO * CD * A * v**2
F_rr   = CRR * MASS * G * np.cos(theta)
F_gr   = MASS * G * np.sin(theta)
F_in   = MASS * a

P_mech_W = (F_aero + F_rr + F_gr + F_in) * v    # W, can be negative

df["P_mech_W"] = P_mech_W

# ==========================================================
# COMPUTE MEASURED BATTERY POWER
# ==========================================================

# Battery power = energy drawn per timestep / timestep duration
df["P_bat_W"] = df["energy"] * 3_600_000 / df["dt_s"]   # W

# ==========================================================
# FILTER TO VALID DRIVING ROWS
# (positive mechanical work, physically plausible battery power)
# ==========================================================

mask = (
    (df["P_bat_W"]  <= P_BAT_MAX_W)  &
    (df["P_bat_W"]  >= P_BAT_MIN_W)  &
    (df["v_mps"]    >  1.0/3.6)       # moving only
)

df_valid = df[mask].copy()
print(f"  Rows used for eta curve: {len(df_valid):,}  "
      f"({len(df_valid)/len(df)*100:.1f}% of total)")

# ==========================================================
# COMPUTE ROW-LEVEL eta_sys = P_mech / P_bat
# ==========================================================

df_valid["eta_sys"] = df_valid["P_mech_W"].abs() / df_valid["P_bat_W"].abs()

# Clip to physically realistic range
# True system efficiency for an EV including all losses: ~30-60%
df_valid["eta_sys"] = df_valid["eta_sys"].clip(0.25, 0.65)

print(f"\n  eta_sys statistics (row level):")
print(f"    mean:   {df_valid['eta_sys'].mean():.4f}")
print(f"    median: {df_valid['eta_sys'].median():.4f}")
print(f"    std:    {df_valid['eta_sys'].std():.4f}")

# ==========================================================
# BIN BY SPEED
# ==========================================================

bins = np.arange(0, 160 + SPEED_BIN_WIDTH, SPEED_BIN_WIDTH)
df_valid["speed_bin"] = pd.cut(df_valid["Speed"], bins)

eta_curve = df_valid.groupby("speed_bin")["eta_sys"].agg(
    eta_powertrain_mean="mean",
    eta_powertrain_p05=lambda x: np.percentile(x, 5),
    eta_powertrain_p95=lambda x: np.percentile(x, 95),
    n_samples="count"
).reset_index()

# Remove underpopulated bins
eta_curve = eta_curve[eta_curve["n_samples"] >= MIN_SAMPLES].copy()

# Bin midpoint as speed
eta_curve["speed_kph"] = eta_curve["speed_bin"].apply(
    lambda x: x.mid if pd.notnull(x) else np.nan
)
eta_curve = eta_curve.drop(columns=["speed_bin"]).dropna(subset=["speed_kph"])

print(f"\n  Speed bins in curve: {len(eta_curve)}")
print(f"  Speed range: {eta_curve['speed_kph'].min():.0f} – "
      f"{eta_curve['speed_kph'].max():.0f} km/h")
print(f"  eta mean range: {eta_curve['eta_powertrain_mean'].min():.4f} – "
      f"{eta_curve['eta_powertrain_mean'].max():.4f}")

# ==========================================================
# OPTIONAL SMOOTHING
# ==========================================================

APPLY_SMOOTHING = True

if APPLY_SMOOTHING and len(eta_curve) >= 5:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothed = lowess(eta_curve["eta_powertrain_mean"],
                      eta_curve["speed_kph"], frac=0.30)
    eta_curve["eta_powertrain_mean"] = smoothed[:, 1]
    print("  LOWESS smoothing applied.")

# ==========================================================
# SANITY CHECK — does this curve recover total energy correctly?
# ==========================================================

eta_interp = np.interp(df["Speed"].values,
                       eta_curve["speed_kph"].values,
                       eta_curve["eta_powertrain_mean"].values,
                       left=eta_curve["eta_powertrain_mean"].iloc[0],
                       right=eta_curve["eta_powertrain_mean"].iloc[-1])

# Combined with battery curve (assumed mean ~0.965)
eta_bat_mean = 0.965
eta_total    = eta_interp * eta_bat_mean

P_mech_all = df["P_mech_W"].values
P_bat_pred = np.where(P_mech_all < 0,
                      P_mech_all * eta_total,
                      P_mech_all / eta_total)

E_pred  = np.nansum(P_bat_pred * df["dt_s"].values / 3_600_000)
E_meas  = df["energy"].sum()
err_pct = (E_pred - E_meas) / abs(E_meas) * 100

print(f"\n  Sanity check — energy recovery:")
print(f"    Predicted : {E_pred:.3f} kWh")
print(f"    Measured  : {E_meas:.3f} kWh")
print(f"    Error     : {err_pct:+.2f}%")
print(f"  (Target: within ±15% — physics model will fine-tune Cd/Crr)")

# ==========================================================
# SAVE
# ==========================================================

eta_curve.to_csv(OUTPUT_FILE, index=False)
print(f"\n  Saved: {OUTPUT_FILE}")
print(eta_curve[["speed_kph","eta_powertrain_mean","n_samples"]].to_string(index=False))

# ==========================================================
# PLOT
# ==========================================================

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=eta_curve["speed_kph"], y=eta_curve["eta_powertrain_p95"],
    mode="lines", line=dict(width=0), showlegend=False
))
fig.add_trace(go.Scatter(
    x=eta_curve["speed_kph"], y=eta_curve["eta_powertrain_p05"],
    mode="lines", fill="tonexty",
    fillcolor="rgba(0,100,255,0.15)", line=dict(width=0),
    name="p05–p95 range"
))
fig.add_trace(go.Scatter(
    x=eta_curve["speed_kph"], y=eta_curve["eta_powertrain_mean"],
    mode="lines+markers", name="Mean eta_sys(v)",
    line=dict(color="royalblue", width=2)
))

fig.update_layout(
    title="True system efficiency curve  η_sys(v)  =  P_mech / P_battery",
    xaxis_title="Speed (km/h)",
    yaxis_title="System efficiency  η_sys",
    yaxis=dict(range=[0.0, 0.80]),
    template="plotly_white"
)
fig.write_image(str(Path("figs") / "eta_powertrain_curve.png"), scale=2)
fig.show()