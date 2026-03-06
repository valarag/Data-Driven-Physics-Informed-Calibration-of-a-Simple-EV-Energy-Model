"""
calibrate_physics_model.py
==========================
STEP 5 — Make the physics model learn its coefficients from the data.

What this script does
---------------------
1. Loads the primary trip (5/4/2022) — same as validation v3.
2. Computes the data-driven mechanical power  P_data = F_res * v
   This is the "teacher" / ground truth.
3. Runs scipy.optimize to find the best values of:
      Cd    — aerodynamic drag coefficient
      Crr   — rolling resistance coefficient
      k_cd  — optional: makes Cd vary linearly with speed  Cd(v) = Cd0 + k_cd*v
   by minimising the RMSE between P_param and P_data.
4. Prints before/after comparison and saves tuned coefficients to
   curves/tuned_physics_params.json
5. Validates the tuned model on ALL THREE trips and prints a summary table.
6. Plots: convergence curve + before/after cumulative energy on all trips.

How to use
----------
Run this AFTER build_eta_powertrain.py and build_eta_battery.py have
already generated their curve CSVs.

    python calibrate_physics_model.py

The tuned coefficients are saved to curves/tuned_physics_params.json.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = "browser"

# ==========================================================
# CONFIGURATION
# ==========================================================

# Trip files
PRIMARY_TRIP_FILE = "Tracking_data_efficiecny.csv"   # used for calibration
CROSS_TRIP_FILES  = [
    "trip_12apr2022.csv",   # unseen trip 1 — validation only
    "trip_11apr2022.csv",   # unseen trip 2 — validation only
]

CURVE_PT_FILE  = Path("curves") / "eta_powertrain_vs_speed.csv"
CURVE_BAT_FILE = Path("curves") / "eta_battery_vs_Tbat.csv"
OUTDIR         = Path("figs");  OUTDIR.mkdir(exist_ok=True)
PARAMS_OUT     = Path("curves") / "tuned_physics_params.json"

# Fixed vehicle parameters (not calibrated — well known)
RHO    = 1.225   # kg/m³  air density
A      = 2.22    # m²     frontal area
MASS   = 1847    # kg
G      = 9.81    # m/s²

# HVAC
IS_AC_ON   = 0
AC_POWER_W = 5500

# Battery temperature assumption
T_BAT_ASSUMED = 10.0

# Initial guesses  [Cd, Crr, k_cd]
#   Cd    — drag coefficient        (Tesla spec: 0.23)
#   Crr   — rolling resistance      (typical road tyre: 0.009)
#   k_cd  — speed-dependent drag    (start at 0 = constant Cd)
X0     = [0.50,  0.030,  0.0005]   # start at current solution
BOUNDS = [(0.20, 1.50),             # Cd effective — can be large
          (0.004, 0.080),           # Crr effective
          (-1e-3, 1e-3)]            # k_cd

REQ_COLS = ["Speed", "DeltaT", "Energy Consumption (kWh)",
            "Total Resistive Force", "Slope Angle (rad)", "Acceleration"]

# ==========================================================
# HELPERS
# ==========================================================

def trip_energy_kwh(df, Cd, Crr, k_cd):
    """Compute total predicted battery energy for a trip."""
    v     = df["v_mps"].to_numpy()
    theta = df["theta"].to_numpy()
    a     = df["accel"].to_numpy()
    eta   = df["eta_sys"].to_numpy()
    dt    = df["dt_s"].to_numpy()

    Cd_v   = Cd + k_cd * v
    F_aero = 0.5 * RHO * Cd_v * A * v**2
    F_rr   = Crr * MASS * G * np.cos(theta)
    F_gr   = MASS * G * np.sin(theta)
    F_in   = MASS * a
    P_mech = (F_aero + F_rr + F_gr + F_in) * v + IS_AC_ON * AC_POWER_W

    P_bat  = np.where(P_mech < 0, P_mech * eta, P_mech / eta)
    return np.sum(P_bat * dt / 3_600_000)

def interp_1d(x, xp, fp):
    return np.interp(np.asarray(x), np.asarray(xp), np.asarray(fp),
                     left=fp.iloc[0], right=fp.iloc[-1])


def load_trip(filepath, pt, bat):
    """Load, clean and attach efficiency curves to a trip DataFrame."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=REQ_COLS).copy()
    df["dt_s"]  = df["DeltaT"].astype(float).clip(0.05, 5.0)
    df["v_mps"] = (df["Speed"] / 3.6).clip(lower=0)
    df["theta"] = df["Slope Angle (rad)"].to_numpy()
    df["accel"] = df["Acceleration"].to_numpy()

    eta_pt  = np.clip(interp_1d(df["Speed"], pt["speed_kph"],
                                pt["eta_powertrain_mean"]), 0.70, 0.98)
    eta_bat = np.clip(interp_1d(np.full(len(df), T_BAT_ASSUMED),
                                bat["T_bat_C"],
                                bat["eta_battery_mean"]), 0.70, 0.995)
    df["eta_sys"] = eta_pt * eta_bat

    # Data-driven mechanical power (teacher signal, allows regen)
    F_res = df["Total Resistive Force"].to_numpy()
    df["P_mech_data_W"] = df["Energy Consumption (kWh)"].to_numpy() * 3_600_000 / df["dt_s"].to_numpy()

    return df


def physics_power(df, Cd, Crr, k_cd):
    """
    Parametric mechanical power for every row.
    Cd(v) = Cd + k_cd * v  allows a gentle speed-dependent correction.
    """
    v     = df["v_mps"].to_numpy()
    theta = df["theta"].to_numpy()
    a     = df["accel"].to_numpy()

    Cd_v   = Cd + k_cd * v
    F_aero = 0.5 * RHO * Cd_v * A * v**2
    F_rr   = Crr * MASS * G * np.cos(theta)
    F_gr   = MASS * G * np.sin(theta)
    F_in   = MASS * a

    return (F_aero + F_rr + F_gr + F_in) * v + IS_AC_ON * AC_POWER_W


def apply_efficiency(P_mech_W, eta_sys):
    """Direction-aware efficiency: discharge draws more, regen recovers less."""
    return np.where(P_mech_W < 0,
                    P_mech_W * eta_sys,
                    P_mech_W / eta_sys)


def cumulative_energy_kwh(P_bat_W, dt_s):
    return np.cumsum(P_bat_W * dt_s / 3600.0) / 1000.0


def trip_errors(df, Cd, Crr, k_cd):
    E_pred  = trip_energy_kwh(df, Cd, Crr, k_cd)
    E_meas  = df["Energy Consumption (kWh)"].sum()
    err_pct = (E_pred - E_meas) / abs(E_meas) * 100
    rmse    = abs(err_pct)   # use energy error as proxy for RMSE display
    return rmse, err_pct

# ==========================================================
# LOAD DATA
# ==========================================================

pt  = pd.read_csv(CURVE_PT_FILE).sort_values("speed_kph")
bat = pd.read_csv(CURVE_BAT_FILE).sort_values("T_bat_C")

print("Loading primary calibration trip ...")
df_primary = load_trip(PRIMARY_TRIP_FILE, pt, bat)
print(f"  Rows: {len(df_primary)}   "
      f"Measured energy: {df_primary['Energy Consumption (kWh)'].sum():.2f} kWh")

cross_trips = {}
for f in CROSS_TRIP_FILES:
    if Path(f).exists():
        cross_trips[f] = load_trip(f, pt, bat)
        print(f"  Cross-trip loaded: {f}  ({len(cross_trips[f])} rows)")
    else:
        print(f"  WARNING: {f} not found — skipping.")

# ==========================================================
# BASELINE  (before calibration)
# ==========================================================

rmse_before, err_before = trip_errors(df_primary, X0[0], X0[1], X0[2])
print(f"\nBEFORE calibration  ->  RMSE: {rmse_before:.3f} kW   "
      f"Energy error: {err_before:+.2f}%")

# ==========================================================
# OPTIMISATION
# ==========================================================

convergence_log = []

def objective(x):
    Cd, Crr, k_cd = x
    E_pred = trip_energy_kwh(df_primary, Cd, Crr, k_cd)
    E_meas = df_primary["Energy Consumption (kWh)"].sum()
    err    = (E_pred - E_meas) / abs(E_meas)
    convergence_log.append(abs(err) * 100)
    return err ** 2

print("\nRunning optimiser (L-BFGS-B) ...")
result = minimize(
    objective,
    x0     = X0,
    bounds = BOUNDS,
    method = "L-BFGS-B",
    options= {"maxiter": 500, "ftol": 1e-10, "gtol": 1e-8}
)

Cd_tuned, Crr_tuned, kcd_tuned = result.x
print(f"Optimiser finished — success: {result.success}  iterations: {result.nit}")

# ==========================================================
# RESULTS
# ==========================================================

rmse_after, err_after = trip_errors(df_primary, Cd_tuned, Crr_tuned, kcd_tuned)

print("\n" + "=" * 60)
print("CALIBRATION RESULTS")
print("=" * 60)
print(f"{'Parameter':<22} {'Before':>10} {'After':>12}")
print(f"{'Cd0':<22} {X0[0]:>10.4f} {Cd_tuned:>12.4f}")
print(f"{'Crr':<22} {X0[1]:>10.4f} {Crr_tuned:>12.4f}")
print(f"{'k_cd (Cd per m/s)':<22} {X0[2]:>10.6f} {kcd_tuned:>12.6f}")
print("-" * 60)
print(f"{'RMSE power (kW)':<22} {rmse_before:>10.3f} {rmse_after:>12.3f}")
print(f"{'Energy error (%)':<22} {err_before:>10.2f} {err_after:>12.2f}")
print("=" * 60)

# ==========================================================
# GENERALISATION TABLE — all three trips
# ==========================================================

print("\n" + "=" * 68)
print("GENERALISATION — tuned model on all trips")
print("=" * 68)
print(f"{'Trip':<22} {'Meas (kWh)':>12} {'Pred (kWh)':>12} {'Error %':>10}")
print("-" * 68)

all_trips = {"Primary  5/4": df_primary}
all_trips.update({k: v for k, v in cross_trips.items()})

for label, df_t in all_trips.items():
    P_p    = physics_power(df_t, Cd_tuned, Crr_tuned, kcd_tuned)
    P_b    = apply_efficiency(P_p, df_t["eta_sys"].to_numpy())
    E_pred = cumulative_energy_kwh(P_b, df_t["dt_s"].to_numpy())[-1]
    E_meas = df_t["Energy Consumption (kWh)"].sum()
    err    = (E_pred - E_meas) / (abs(E_meas) + 1e-9) * 100
    print(f"  {label:<20} {E_meas:>12.3f} {E_pred:>12.3f} {err:>+9.2f}%")

print("=" * 68)

# ==========================================================
# SAVE TUNED PARAMETERS
# ==========================================================

tuned_params = {
    "Cd0"  : float(Cd_tuned),
    "Crr"  : float(Crr_tuned),
    "k_cd" : float(kcd_tuned),
    "RHO"  : RHO, "A": A, "MASS": MASS, "G": G,
    "note" : "Calibrated by calibrate_physics_model.py via L-BFGS-B"
}
PARAMS_OUT.parent.mkdir(exist_ok=True)
with open(PARAMS_OUT, "w") as fh:
    json.dump(tuned_params, fh, indent=2)
print(f"\nTuned parameters saved -> {PARAMS_OUT}")

# ==========================================================
# PLOTS
# ==========================================================

# --- Plot 1: Optimiser convergence ---
fig1 = go.Figure()
fig1.add_trace(go.Scatter(y=convergence_log, mode="lines", name="RMSE (kW)"))
fig1.update_layout(title="Optimiser convergence — RMSE vs iteration",
                   xaxis_title="Iteration", yaxis_title="RMSE (kW)",
                   template="plotly_white")
fig1.write_image(str(OUTDIR / "calibration_convergence.png"), scale=2)
fig1.show()

# --- Plot 2: Before vs After cumulative energy (primary trip) ---
P_bat_before = apply_efficiency(
    physics_power(df_primary, X0[0], X0[1], X0[2]),
    df_primary["eta_sys"].to_numpy())
P_bat_after  = apply_efficiency(
    physics_power(df_primary, Cd_tuned, Crr_tuned, kcd_tuned),
    df_primary["eta_sys"].to_numpy())

E_meas_cum  = np.cumsum(df_primary["Energy Consumption (kWh)"].to_numpy())
E_before_cum = cumulative_energy_kwh(P_bat_before, df_primary["dt_s"].to_numpy())
E_after_cum  = cumulative_energy_kwh(P_bat_after,  df_primary["dt_s"].to_numpy())

fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=E_meas_cum,    mode="lines", name="Measured",      line=dict(color="black")))
fig2.add_trace(go.Scatter(y=E_before_cum,  mode="lines", name="Before calib.", line=dict(color="red",   dash="dash")))
fig2.add_trace(go.Scatter(y=E_after_cum,   mode="lines", name="After calib.",  line=dict(color="green")))
fig2.update_layout(title="Cumulative energy — before vs after calibration (primary trip)",
                   xaxis_title="Sample index", yaxis_title="Energy (kWh)",
                   template="plotly_white")
fig2.write_image(str(OUTDIR / "calibration_energy_primary.png"), scale=2)
fig2.show()

# --- Plot 3: Tuned model on ALL trips ---
n = len(all_trips)
fig3 = make_subplots(rows=n, cols=1, shared_xaxes=False,
                     vertical_spacing=0.08,
                     subplot_titles=list(all_trips.keys()))

for i, (label, df_t) in enumerate(all_trips.items(), start=1):
    P_p = apply_efficiency(physics_power(df_t, Cd_tuned, Crr_tuned, kcd_tuned),
                           df_t["eta_sys"].to_numpy())
    E_p = cumulative_energy_kwh(P_p, df_t["dt_s"].to_numpy())
    E_m = np.cumsum(df_t["Energy Consumption (kWh)"].to_numpy())
    show = (i == 1)
    fig3.add_trace(go.Scatter(y=E_m, mode="lines", name="Measured",
                              line=dict(color="black"), showlegend=show), row=i, col=1)
    fig3.add_trace(go.Scatter(y=E_p, mode="lines", name="Tuned parametric",
                              line=dict(color="green"), showlegend=show), row=i, col=1)
    fig3.update_yaxes(title_text="Energy (kWh)", row=i, col=1)

fig3.update_layout(height=350*n,
                   title_text="Tuned parametric model — all trips",
                   template="plotly_white")
fig3.write_image(str(OUTDIR / "calibration_all_trips.png"), scale=2)
fig3.show()

# --- Plot 4: Instantaneous power before/after (primary trip, downsampled) ---
step = max(1, len(df_primary) // 8000)
idx  = np.arange(len(df_primary))[::step]
P_meas_kW = (df_primary["Energy Consumption (kWh)"].to_numpy()
             / df_primary["dt_s"].to_numpy() * 3600.0)[::step]

fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=idx, y=P_meas_kW,
                          mode="lines", name="Measured", line=dict(color="black")))
fig4.add_trace(go.Scatter(x=idx, y=P_bat_before[::step]/1000.0,
                          mode="lines", name="Before calib.", line=dict(color="red", dash="dash")))
fig4.add_trace(go.Scatter(x=idx, y=P_bat_after[::step]/1000.0,
                          mode="lines", name="After calib.", line=dict(color="green")))
fig4.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8)
fig4.update_layout(title="Instantaneous power — before vs after calibration",
                   xaxis_title="Sample index (downsampled)", yaxis_title="Power (kW)",
                   template="plotly_white")
fig4.write_image(str(OUTDIR / "calibration_power_instant.png"), scale=2)
fig4.show()

print("\nAll figures saved to figs/")
print(f"Tuned: Cd={Cd_tuned:.4f}  Crr={Crr_tuned:.4f}  k_cd={kcd_tuned:.6f}")
