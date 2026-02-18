import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"  # avoids nbformat error

matplotlib_mode = True


# =========================
# CONFIG (edit if needed)
# =========================
TRACKING_FILE = "Tracking_data_efficiecny.csv"
CURVE_PT_FILE = Path("curves") / "eta_powertrain_vs_speed.csv"
CURVE_BAT_FILE = Path("curves") / "eta_battery_vs_Tbat.csv"

OUTDIR = Path("figs")
OUTDIR.mkdir(exist_ok=True)

# Vehicle parameters (assumptions; keep consistent in report)
RHO = 1.225           # air density kg/m3
CD = 0.29
A = 2.3               # frontal area m2
CRR = 0.011
MASS = 1350           # kg
G = 9.81

# sanity
EPS_V = 0.1           # m/s to avoid divide issues

# =========================
# LOAD CURVES & BUILD INTERPOLATORS
# =========================
pt = pd.read_csv(CURVE_PT_FILE)
bat = pd.read_csv(CURVE_BAT_FILE)

# sort for interpolation
pt = pt.sort_values("speed_kph")
bat = bat.sort_values("T_bat_C")

def interp_1d(x, xp, fp, left=None, right=None):
    """Safe 1D linear interpolation."""
    return np.interp(x, xp, fp,
                     left=fp.iloc[0] if left is None else left,
                     right=fp.iloc[-1] if right is None else right)

# =========================
# LOAD TRACKING DATA
# =========================
df = pd.read_csv(TRACKING_FILE)

# required columns (as seen in your exploration)
# Speed (km/h), Acceleration (m/s^2), Slope Angle (rad), DeltaT (s), Energy Consumption (kWh)
required = ["Speed", "Acceleration", "Slope Angle (rad)", "DeltaT", "Energy Consumption (kWh)"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in tracking file: {missing}")

df = df.dropna(subset=required).copy()

# Speed conversion
df["v_mps"] = df["Speed"] / 3.6
df["v_mps"] = df["v_mps"].clip(lower=0)

# Time step (DeltaT can be weird sometimes; clamp)
df["dt_s"] = df["DeltaT"].astype(float)
df["dt_s"] = df["dt_s"].clip(lower=0.05, upper=5.0)  # avoid insane outliers

# =========================
# PHYSICS POWER TERMS
# =========================
v = df["v_mps"].to_numpy()
a = df["Acceleration"].to_numpy()
theta = df["Slope Angle (rad)"].to_numpy()

# Forces
F_aero = 0.5 * RHO * CD * A * v**2
F_rr   = CRR * MASS * G * np.cos(theta)
F_gr   = MASS * G * np.sin(theta)
F_in   = MASS * a

# Traction power (W)
P_mech_W = (F_aero + F_rr + F_gr + F_in) * v

# Remove negative mech power for this first validation (regen not modeled here)
P_mech_W = np.maximum(P_mech_W, 0.0)

df["P_mech_kW"] = P_mech_W / 1000

# =========================
# APPLY CALIBRATED EFFICIENCIES
# =========================
# η_powertrain(v): use speed in kph
eta_pt = interp_1d(df["Speed"].to_numpy(), pt["speed_kph"], pt["eta_powertrain_mean"])
eta_pt = np.clip(eta_pt, 0.70, 0.98)

# Battery temperature is NOT present in tracking dataset.
# So we use a reasonable constant (25°C) for now OR you can map ambient if available.
T_assumed = 25.0
eta_bat = interp_1d(np.full(len(df), T_assumed), bat["T_bat_C"], bat["eta_battery_mean"])
eta_bat = np.clip(eta_bat, 0.70, 0.995)

df["eta_pt"] = eta_pt
df["eta_bat"] = eta_bat

# Predicted electrical power at battery (kW)
# P_bat_pred = P_mech / (eta_pt * eta_bat)
df["P_bat_pred_kW"] = df["P_mech_kW"] / (df["eta_pt"] * df["eta_bat"])

# =========================
# INTEGRATE ENERGY
# =========================
# Predicted energy (kWh)
df["E_pred_kWh_cum"] = np.cumsum(df["P_bat_pred_kW"] * df["dt_s"] / 3600.0)

# Measured energy: dataset gives "Energy Consumption (kWh)" per sample (likely incremental)
# We will treat it as incremental and accumulate.
df["E_meas_kWh_cum"] = np.cumsum(df["Energy Consumption (kWh)"] * 1.0)

E_pred = df["E_pred_kWh_cum"].iloc[-1]
E_meas = df["E_meas_kWh_cum"].iloc[-1]
rel_err = (E_pred - E_meas) / (E_meas + 1e-9) * 100

print("\n================ VALIDATION SUMMARY ================")
print(f"Samples: {len(df)}")
print(f"Total measured energy (kWh):  {E_meas:.3f}")
print(f"Total predicted energy (kWh): {E_pred:.3f}")
print(f"Relative error (%):           {rel_err:.2f}")
print("====================================================\n")

# =========================
# PLOTS
# =========================
if matplotlib_mode:
    # 1) Cumulative energy
    plt.figure()
    plt.plot(df["E_meas_kWh_cum"].to_numpy(), label="Measured cumulative energy (kWh)")
    plt.plot(df["E_pred_kWh_cum"].to_numpy(), label="Predicted cumulative energy (kWh)")
    plt.title("Cumulative Energy: Measured vs Predicted")
    plt.xlabel("Sample index")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "energy_cum_meas_vs_pred.png", dpi=200)
    plt.show()

    # 2) Power comparison (downsample for readability)
    step = max(1, len(df)//5000)
    plt.figure()
    plt.plot(df["P_mech_kW"].to_numpy()[::step], label="Mechanical power (kW)")
    plt.plot(df["P_bat_pred_kW"].to_numpy()[::step], label="Predicted battery power (kW)")
    plt.title("Power (downsampled): Mechanical vs Predicted Battery")
    plt.xlabel("Sample index (downsampled)")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "power_mech_vs_pred_bat.png", dpi=200)
    plt.show()
else:
    # 1) Cumulative energy
    fig1 = go.Figure()

    fig1.add_trace(
        go.Scatter(
            y=df["E_meas_kWh_cum"].to_numpy(),
            mode="lines",
            name="Measured cumulative energy (kWh)"
        )
    )

    fig1.add_trace(
        go.Scatter(
            y=df["E_pred_kWh_cum"].to_numpy(),
            mode="lines",
            name="Predicted cumulative energy (kWh)"
        )
    )

    fig1.update_layout(
        title="Cumulative Energy: Measured vs Predicted",
        xaxis_title="Sample index",
        yaxis_title="Energy (kWh)",
        template="plotly_white"
    )

    fig1.write_image(str(OUTDIR / "energy_cum_meas_vs_pred.png"), scale=2)
    fig1.show()


    # 2) Power comparison (downsample for readability)
    step = max(1, len(df)//5000)

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            y=df["P_mech_kW"].to_numpy()[::step],
            mode="lines",
            name="Mechanical power (kW)"
        )
    )

    fig2.add_trace(
        go.Scatter(
            y=df["P_bat_pred_kW"].to_numpy()[::step],
            mode="lines",
            name="Predicted battery power (kW)"
        )
    )

    fig2.update_layout(
        title="Power (downsampled): Mechanical vs Predicted Battery",
        xaxis_title="Sample index (downsampled)",
        yaxis_title="Power (kW)",
        template="plotly_white"
    )

    fig2.write_image(str(OUTDIR / "power_mech_vs_pred_bat.png"), scale=2)
    fig2.show()
