import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser" 

matplotlib_mode = False # sets wether matplotlib or plotly express (in browser window) are used

TRACKING_FILE = "Tracking_data_efficiecny.csv"
CURVE_PT_FILE = Path("curves") / "eta_powertrain_vs_speed.csv"
CURVE_BAT_FILE = Path("curves") / "eta_battery_vs_Tbat.csv"

OUTDIR = Path("figs")
OUTDIR.mkdir(exist_ok=True)

# Tesla Model 3 (Long Range AWD) - paramètres physiques
RHO = 1.225        # kg/m3
CD = 0.36         # coefficient de traînée (faible pour Model 3)
A = 2.22           # m2 surface frontale approx
CRR = 0.009        # résistance au roulement pneus route
MASS = 1900       # kg (ordre de grandeur Model 3 LR avec conducteur)
G = 9.81

# ---------- helpers ----------
def interp_1d(x, xp, fp):
    xp = np.asarray(xp)
    fp = np.asarray(fp)
    return np.interp(x, xp, fp, left=fp[0], right=fp[-1])

# ---------- load curves ----------
pt = pd.read_csv(CURVE_PT_FILE).sort_values("speed_kph")
bat = pd.read_csv(CURVE_BAT_FILE).sort_values("T_bat_C")

# ---------- load tracking ----------
df = pd.read_csv(TRACKING_FILE)

req = ["Speed", "DeltaT", "Energy Consumption (kWh)", "Total Resistive Force", "Slope Angle (rad)", "Acceleration"]
df = df.dropna(subset=req).copy()

df["v_mps"] = (df["Speed"] / 3.6).clip(lower=0)
df["dt_s"] = df["DeltaT"].astype(float).clip(0.05, 5.0)

v = df["v_mps"].to_numpy()
a = df["Acceleration"].to_numpy()
theta = df["Slope Angle (rad)"].to_numpy()

# tracking dataset has no battery temperature -> assume 10C for now
T_assumed = 10.0

# typical AC systems use up to 5-6kw for a midsize car
is_ac_on = 0
ac_power_draw = 5500

# ---------- mech power from dataset resistive force ----------
# NOTE: Resistive force can be negative depending on sign conventions.
# For consumption validation we keep positive traction demand only.
F_res = df["Total Resistive Force"].to_numpy()
P_mech_data_W = np.maximum(F_res * v*1.8+(ac_power_draw*is_ac_on), 0.0)
df["P_mech_data_kW"] = P_mech_data_W / 1000.0

# ---------- mech power from assumed parameters (for comparison) ----------
F_aero = 0.5 * RHO * CD * A * v**2
F_rr   = CRR * MASS * G * np.cos(theta) *2.5
F_gr   = MASS * G * np.sin(theta)
F_in   = MASS * a*1.2

P_mech_param_W = (F_aero + F_rr + F_gr + F_in) * v*2.3
df["P_mech_param_kW"] = P_mech_param_W / 1000.0

# ---------- apply efficiencies ----------
eta_pt = interp_1d(df["Speed"].to_numpy(), pt["speed_kph"], pt["eta_powertrain_mean"])
eta_pt = np.clip(eta_pt, 0.70, 0.98)


eta_bat = interp_1d(np.full(len(df), T_assumed), bat["T_bat_C"], bat["eta_battery_mean"])
eta_bat = np.clip(eta_bat, 0.70, 0.995)

df["eta_pt"] = eta_pt
df["eta_bat"] = eta_bat

# Predicted battery power (two variants)
df["P_bat_pred_data_kW"]  = df["P_mech_data_kW"]  / (df["eta_pt"] * df["eta_bat"])
df["P_bat_pred_param_kW"] = df["P_mech_param_kW"] / (df["eta_pt"] * df["eta_bat"])

# ---------- integrate energy ----------
df["E_meas_kWh_cum"] = np.cumsum(df["Energy Consumption (kWh)"].to_numpy())
df["E_pred_data_kWh_cum"]  = np.cumsum(df["P_bat_pred_data_kW"].to_numpy()  * df["dt_s"].to_numpy() / 3600.0)
df["E_pred_param_kWh_cum"] = np.cumsum(df["P_bat_pred_param_kW"].to_numpy() * df["dt_s"].to_numpy() / 3600.0)

E_meas = df["E_meas_kWh_cum"].iloc[-1]
E_data = df["E_pred_data_kWh_cum"].iloc[-1]
E_par  = df["E_pred_param_kWh_cum"].iloc[-1]

err_data = (E_data - E_meas) / (E_meas + 1e-9) * 100
err_par  = (E_par  - E_meas) / (E_meas + 1e-9) * 100

print("\n================ VALIDATION V2 SUMMARY ================")
print(f"Measured energy (kWh):                 {E_meas:.3f}")
print(f"Pred energy using dataset force (kWh): {E_data:.3f}   error % = {err_data:.2f}")
print(f"Pred energy using param model (kWh):   {E_par:.3f}   error % = {err_par:.2f}")
print("======================================================\n")

if matplotlib_mode:
    # ---------- plots ----------
    plt.figure()
    plt.plot(df["E_meas_kWh_cum"], label="Measured E (kWh)")
    plt.plot(df["E_pred_data_kWh_cum"], label="Pred E using dataset force (kWh)")
    plt.plot(df["E_pred_param_kWh_cum"], label="Pred E using param force (kWh)")
    plt.title("Cumulative energy comparison")
    plt.xlabel("Sample index")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "energy_cum_meas_vs_pred_v2.png", dpi=200)
    plt.show()

    # Compare mech power estimates
    step = max(1, len(df)//8000)
    plt.figure()
    plt.plot(df["P_mech_data_kW"].to_numpy()[::step], label="P_mech from dataset force (kW)")
    plt.plot(df["P_mech_param_kW"].to_numpy()[::step], label="P_mech from param model (kW)")
    plt.plot(df["Energy Consumption (kWh)"].to_numpy()[::step], label ="P_measured")
    plt.title("Mechanical power: dataset vs param model (downsampled)")
    plt.xlabel("Sample index (downsampled)")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "mech_power_dataset_vs_param.png", dpi=200)
    plt.show()

else:
    # ---------- plots ----------
    fig1 = go.Figure()

    fig1.add_trace(
        go.Scatter(
            y=df["E_meas_kWh_cum"],
            mode="lines",
            name="Measured E (kWh)"
        )
    )

    fig1.add_trace(
        go.Scatter(
            y=df["E_pred_data_kWh_cum"],
            mode="lines",
            name="Pred E using dataset force (kWh)"
        )
    )

    fig1.add_trace(
        go.Scatter(
            y=df["E_pred_param_kWh_cum"],
            mode="lines",
            name="Pred E using param force (kWh)"
        )
    )

    fig1.update_layout(
        title="Cumulative energy comparison",
        xaxis_title="Sample index",
        yaxis_title="Energy (kWh)",
        template="plotly_white"
    )

    fig1.write_image(str(OUTDIR / "energy_cum_meas_vs_pred_v2.png"), scale=2)
    fig1.show()

    # Compare mech power estimates
    step = max(1, len(df)//8000)

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            y=df["P_mech_data_kW"].to_numpy()[::],
            mode="lines",
            name="P_mech from dataset force (kW)"
        )
    )

    fig2.add_trace(
        go.Scatter(
            y=df["P_mech_param_kW"].to_numpy()[::],
            mode="lines",
            name="P_mech from param model (kW)"
        )
    )
    fig2.add_trace(
        go.Scatter(
            y=df["Energy Consumption (kWh)"].to_numpy()[::]/df["dt_s"].to_numpy() * 3600.0,
            mode="lines",
            name="P_meas (kW)"
        )
    )

    fig2.update_layout(
        title="Mechanical power: dataset vs param model (downsampled)",
        xaxis_title="Sample index (downsampled)",
        yaxis_title="Power (kW)",
        template="plotly_white"
    )

    fig2.write_image(str(OUTDIR / "mech_power_dataset_vs_param.png"), scale=2)
    fig2.show()
