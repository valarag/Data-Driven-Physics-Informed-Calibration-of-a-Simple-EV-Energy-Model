import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = "browser"

matplotlib_mode = False  # True = matplotlib, False = Plotly in browser

# ==========================================================
# FILES
# ==========================================================

TRACKING_FILE = "Tracking_data_efficiecny.csv"   # primary trip (same as v2)
CURVE_PT_FILE  = Path("curves") / "eta_powertrain_vs_speed.csv"
CURVE_BAT_FILE = Path("curves") / "eta_battery_vs_Tbat.csv"

# ── Cross-trip validation ──────────────────────────────────
# Set this to a DIFFERENT trip CSV to test generalisation.
# Must have the same columns as TRACKING_FILE.
# Leave as None to skip cross-trip validation.
CROSS_TRIP_FILE = None   # e.g. "Tracking_data_trip2.csv"

OUTDIR = Path("figs")
OUTDIR.mkdir(exist_ok=True)

# ==========================================================
# PHYSICAL PARAMETERS  (Tesla Model 3 LR AWD)
# ==========================================================

RHO  = 1.225    # air density  kg/m³
CD   = 0.23     # drag coefficient
A    = 2.22     # frontal area  m²
CRR  = 0.009    # rolling resistance coefficient
MASS = 1847     # kg  (vehicle + driver)
G    = 9.81     # m/s²

# HVAC / auxiliaries
IS_AC_ON     = 0       # 0 = off, 1 = on
AC_POWER_W   = 5500    # W  (only used when IS_AC_ON = 1)

# Battery temperature assumption (tracking file has no T_bat column)
T_BAT_ASSUMED = 10.0   # °C

# ==========================================================
# HELPERS
# ==========================================================

def interp_1d(x, xp, fp):
    """1-D linear interpolation with edge clamping."""
    return np.interp(np.asarray(x), np.asarray(xp), np.asarray(fp),
                     left=fp.iloc[0], right=fp.iloc[-1])


def compute_model(df, pt, bat):
    """
    Core model: given a trip DataFrame, return a copy with all
    predicted power and energy columns added.

    FIX vs v2
    ---------
    * Regenerative braking is now modelled:
        - np.maximum(..., 0) floors are REMOVED from both power paths.
        - When force is negative (regen / downhill), power goes negative.
        - Efficiency is applied differently for regen:
              discharge:  P_bat = P_mech / (eta_pt * eta_bat)   [draws more from battery]
              regen:      P_bat = P_mech * (eta_pt * eta_bat)   [recovers less to battery]
      This matches real inverter / motor behaviour.
    """
    df = df.copy()

    v     = (df["Speed"] / 3.6).to_numpy()          # m/s  (always >= 0)
    a     = df["Acceleration"].to_numpy()
    theta = df["Slope Angle (rad)"].to_numpy()
    dt    = df["dt_s"].to_numpy()

    # ── Efficiency curves ──────────────────────────────────
    eta_pt  = np.clip(interp_1d(df["Speed"], pt["speed_kph"],  pt["eta_powertrain_mean"]),  0.70, 0.98)
    eta_bat = np.clip(interp_1d(np.full(len(df), T_BAT_ASSUMED), bat["T_bat_C"], bat["eta_battery_mean"]), 0.70, 0.995)
    eta_sys = eta_pt * eta_bat   # combined system efficiency

    df["eta_pt"]  = eta_pt
    df["eta_bat"] = eta_bat

    # ──────────────────────────────────────────────────────
    # PATH 1 – dataset resistive force
    # ──────────────────────────────────────────────────────
    F_res = df["Total Resistive Force"].to_numpy()
    P_mech_data_W = F_res * v + IS_AC_ON * AC_POWER_W   # can be negative (regen)

    # Apply efficiency: direction-aware
    regen_mask = P_mech_data_W < 0
    P_bat_data_W = np.where(
        regen_mask,
        P_mech_data_W * eta_sys,        # regen: recover less
        P_mech_data_W / eta_sys         # drive:  draw more
    )

    df["P_mech_data_kW"] = P_mech_data_W / 1000.0
    df["P_bat_data_kW"]  = P_bat_data_W  / 1000.0

    # ──────────────────────────────────────────────────────
    # PATH 2 – parametric physics model
    # ──────────────────────────────────────────────────────
    F_aero = 0.5 * RHO * CD * A * v**2              # always >= 0
    F_rr   = CRR * MASS * G * np.cos(theta)         # always >= 0
    F_gr   = MASS * G * np.sin(theta)               # sign from slope
    F_in   = MASS * a                               # sign from acceleration

    # Total traction force (can be negative on downhill / braking)
    F_total_param = F_aero + F_rr + F_gr + F_in
    P_mech_param_W = F_total_param * v + IS_AC_ON * AC_POWER_W

    regen_mask_param = P_mech_param_W < 0
    P_bat_param_W = np.where(
        regen_mask_param,
        P_mech_param_W * eta_sys,
        P_mech_param_W / eta_sys
    )

    df["P_mech_param_kW"] = P_mech_param_W / 1000.0
    df["P_bat_param_kW"]  = P_bat_param_W  / 1000.0

    # ──────────────────────────────────────────────────────
    # Cumulative energy  (kWh)
    # ──────────────────────────────────────────────────────
    df["E_meas_kWh_cum"]  = np.cumsum(df["Energy Consumption (kWh)"].to_numpy())
    df["E_data_kWh_cum"]  = np.cumsum(df["P_bat_data_kW"].to_numpy()  * dt / 3600.0)
    df["E_param_kWh_cum"] = np.cumsum(df["P_bat_param_kW"].to_numpy() * dt / 3600.0)

    return df


def print_summary(df, label=""):
    E_meas  = df["E_meas_kWh_cum"].iloc[-1]
    E_data  = df["E_data_kWh_cum"].iloc[-1]
    E_param = df["E_param_kWh_cum"].iloc[-1]

    err_data  = (E_data  - E_meas) / (abs(E_meas) + 1e-9) * 100
    err_param = (E_param - E_meas) / (abs(E_meas) + 1e-9) * 100

    tag = f"  [{label}]" if label else ""
    print(f"\n================ VALIDATION V3 SUMMARY{tag} ================")
    print(f"  Measured energy            : {E_meas:.3f} kWh")
    print(f"  Pred (dataset force)       : {E_data:.3f} kWh   error = {err_data:+.2f}%")
    print(f"  Pred (parametric model)    : {E_param:.3f} kWh   error = {err_param:+.2f}%")
    print("=" * 56)


# ==========================================================
# LOAD DATA
# ==========================================================

pt  = pd.read_csv(CURVE_PT_FILE).sort_values("speed_kph")
bat = pd.read_csv(CURVE_BAT_FILE).sort_values("T_bat_C")

req = ["Speed", "DeltaT", "Energy Consumption (kWh)",
       "Total Resistive Force", "Slope Angle (rad)", "Acceleration"]

df_raw = pd.read_csv(TRACKING_FILE)
df_raw = df_raw.dropna(subset=req).copy()
df_raw["dt_s"] = df_raw["DeltaT"].astype(float).clip(0.05, 5.0)

df = compute_model(df_raw, pt, bat)
print_summary(df, label="Primary trip")

# ==========================================================
# CROSS-TRIP VALIDATION  (optional second trip)
# ==========================================================

df_cross = None
if CROSS_TRIP_FILE is not None:
    df_cross_raw = pd.read_csv(CROSS_TRIP_FILE)
    df_cross_raw = df_cross_raw.dropna(subset=req).copy()
    df_cross_raw["dt_s"] = df_cross_raw["DeltaT"].astype(float).clip(0.05, 5.0)
    df_cross = compute_model(df_cross_raw, pt, bat)
    print_summary(df_cross, label="Cross-trip")


# ==========================================================
# PLOTTING
# ==========================================================

step = max(1, len(df) // 8000)   # downsample for large files


# ----------------------------------------------------------
# PLOT 1 – Cumulative energy comparison  (same as v2 + regen)
# ----------------------------------------------------------

if matplotlib_mode:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["E_meas_kWh_cum"].values,  label="Measured")
    ax.plot(df["E_data_kWh_cum"].values,  label="Pred – dataset force")
    ax.plot(df["E_param_kWh_cum"].values, label="Pred – parametric")
    ax.set_title("Cumulative energy comparison  (with regen)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Energy (kWh)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "energy_cum_v3.png", dpi=200)
    plt.show()

else:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=df["E_meas_kWh_cum"],  mode="lines", name="Measured"))
    fig1.add_trace(go.Scatter(y=df["E_data_kWh_cum"],  mode="lines", name="Pred – dataset force"))
    fig1.add_trace(go.Scatter(y=df["E_param_kWh_cum"], mode="lines", name="Pred – parametric"))
    fig1.update_layout(
        title="Cumulative energy comparison  (with regen)",
        xaxis_title="Sample index",
        yaxis_title="Energy (kWh)",
        template="plotly_white"
    )
    fig1.write_image(str(OUTDIR / "energy_cum_v3.png"), scale=2)
    fig1.show()


# ----------------------------------------------------------
# PLOT 2 – Instantaneous battery power  (with regen, downsampled)
# ── This is the power plot the tutor asked to compare peaks ──
# ----------------------------------------------------------

idx = np.arange(len(df))[::step]
P_meas_kW  = (df["Energy Consumption (kWh)"].to_numpy() / df["dt_s"].to_numpy() * 3600.0)[::step]

if matplotlib_mode:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(idx, P_meas_kW,                        label="Measured power (kW)")
    ax.plot(idx, df["P_bat_data_kW"].values[::step],  label="Pred – dataset force (kW)")
    ax.plot(idx, df["P_bat_param_kW"].values[::step], label="Pred – parametric (kW)")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_title("Instantaneous battery power  (negative = regen)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Power (kW)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "power_instant_v3.png", dpi=200)
    plt.show()

else:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=idx, y=P_meas_kW,                        mode="lines", name="Measured power"))
    fig2.add_trace(go.Scatter(x=idx, y=df["P_bat_data_kW"].values[::step],  mode="lines", name="Pred – dataset force"))
    fig2.add_trace(go.Scatter(x=idx, y=df["P_bat_param_kW"].values[::step], mode="lines", name="Pred – parametric"))
    fig2.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8)
    fig2.update_layout(
        title="Instantaneous battery power  (negative = regen)",
        xaxis_title="Sample index",
        yaxis_title="Power (kW)",
        template="plotly_white"
    )
    fig2.write_image(str(OUTDIR / "power_instant_v3.png"), scale=2)
    fig2.show()


# ----------------------------------------------------------
# PLOT 3 – DIAGNOSTIC multi-axis plot
# ── Tutor explicitly requested this ──
# Shows: battery power (both models + measured) overlaid with
#        speed, slope angle, acceleration on secondary axes.
# ----------------------------------------------------------

if matplotlib_mode:
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    # Row 0 – battery power
    ax0 = axes[0]
    ax0.plot(idx, P_meas_kW,                           label="Measured power")
    ax0.plot(idx, df["P_bat_data_kW"].values[::step],  label="Pred – dataset force")
    ax0.plot(idx, df["P_bat_param_kW"].values[::step], label="Pred – parametric")
    ax0.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax0.set_ylabel("Battery power (kW)")
    ax0.set_title("Diagnostic: power peaks vs. driving inputs")
    ax0.legend(fontsize=8)

    # Row 1 – speed
    ax1 = axes[1]
    ax1.plot(idx, df["Speed"].values[::step], color="darkorange")
    ax1.set_ylabel("Speed (km/h)")

    # Row 2 – slope angle
    ax2 = axes[2]
    slope_deg = np.degrees(df["Slope Angle (rad)"].values[::step])
    ax2.plot(idx, slope_deg, color="green")
    ax2.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax2.set_ylabel("Slope (°)")

    # Row 3 – acceleration
    ax3 = axes[3]
    ax3.plot(idx, df["Acceleration"].values[::step], color="red")
    ax3.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax3.set_ylabel("Acceleration (m/s²)")
    ax3.set_xlabel("Sample index (downsampled)")

    plt.tight_layout()
    plt.savefig(OUTDIR / "diagnostic_multiaxis_v3.png", dpi=200)
    plt.show()

else:
    fig3 = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            "Battery power (kW)  — negative = regen",
            "Speed (km/h)",
            "Slope angle (°)",
            "Acceleration (m/s²)"
        ]
    )

    # Row 1 – battery power
    fig3.add_trace(go.Scatter(x=idx, y=P_meas_kW,                        mode="lines", name="Measured power"),   row=1, col=1)
    fig3.add_trace(go.Scatter(x=idx, y=df["P_bat_data_kW"].values[::step],  mode="lines", name="Pred – dataset force"), row=1, col=1)
    fig3.add_trace(go.Scatter(x=idx, y=df["P_bat_param_kW"].values[::step], mode="lines", name="Pred – parametric"),    row=1, col=1)
    fig3.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8, row=1, col=1)

    # Row 2 – speed
    fig3.add_trace(
        go.Scatter(x=idx, y=df["Speed"].values[::step], mode="lines",
                   name="Speed (km/h)", line=dict(color="darkorange")),
        row=2, col=1
    )

    # Row 3 – slope
    slope_deg = np.degrees(df["Slope Angle (rad)"].values[::step])
    fig3.add_trace(
        go.Scatter(x=idx, y=slope_deg, mode="lines",
                   name="Slope (°)", line=dict(color="green")),
        row=3, col=1
    )
    fig3.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8, row=3, col=1)

    # Row 4 – acceleration
    fig3.add_trace(
        go.Scatter(x=idx, y=df["Acceleration"].values[::step], mode="lines",
                   name="Acceleration (m/s²)", line=dict(color="red")),
        row=4, col=1
    )
    fig3.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8, row=4, col=1)

    fig3.update_layout(
        height=900,
        title_text="Diagnostic: power peaks vs. driving inputs",
        template="plotly_white"
    )
    fig3.update_xaxes(title_text="Sample index (downsampled)", row=4, col=1)

    fig3.write_image(str(OUTDIR / "diagnostic_multiaxis_v3.png"), scale=2)
    fig3.show()


# ----------------------------------------------------------
# PLOT 4 – Cross-trip validation  (only if second trip loaded)
# ----------------------------------------------------------

if df_cross is not None:
    step_c = max(1, len(df_cross) // 8000)

    if matplotlib_mode:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_cross["E_meas_kWh_cum"].values,  label="Measured")
        ax.plot(df_cross["E_data_kWh_cum"].values,  label="Pred – dataset force")
        ax.plot(df_cross["E_param_kWh_cum"].values, label="Pred – parametric")
        ax.set_title("Cross-trip validation – cumulative energy")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Energy (kWh)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTDIR / "cross_trip_energy_v3.png", dpi=200)
        plt.show()

    else:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(y=df_cross["E_meas_kWh_cum"],  mode="lines", name="Measured"))
        fig4.add_trace(go.Scatter(y=df_cross["E_data_kWh_cum"],  mode="lines", name="Pred – dataset force"))
        fig4.add_trace(go.Scatter(y=df_cross["E_param_kWh_cum"], mode="lines", name="Pred – parametric"))
        fig4.update_layout(
            title="Cross-trip validation – cumulative energy",
            xaxis_title="Sample index",
            yaxis_title="Energy (kWh)",
            template="plotly_white"
        )
        fig4.write_image(str(OUTDIR / "cross_trip_energy_v3.png"), scale=2)
        fig4.show()
