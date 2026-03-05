"""
build_trip_dataset.py
=====================
Phase 1 — Step 1 of the ML pipeline (Weeks 4–6)

What this script does
---------------------
1.  Loads the primary calibration trip (Tracking_data_efficiecny.csv).
2.  Cleans the data: clips DeltaT outliers (sensor glitches), removes
    rows with implausible values.
3.  Segments the continuous recording into individual sub-trips by
    detecting stops longer than STOP_THRESHOLD_S seconds.
4.  Computes one row of features per sub-trip:
        avg_speed_kph, dist_km, duration_min, elev_gain_m,
        max_slope_deg, avg_slope_deg, energy_kWh,
        efficiency_kWh_100km, regen_fraction
5.  Adds the calibrated physics model prediction per sub-trip
    (physics_pred_kWh) and computes the residual:
        residual_kWh = energy_kWh - physics_pred_kWh
    This residual is what XGBoost will learn to correct in hybrid_model.py.
6.  Saves the trip-level ML training table to  data/trip_features.csv
7.  Produces four EDA plots saved to  figs/eda_*.png:
        - Correlation heatmap of all numeric features
        - Efficiency vs average speed (scatter)
        - Efficiency vs elevation gain (scatter)
        - Residual distribution (histogram) — is physics bias consistent?

How to run
----------
    python build_trip_dataset.py

Dependencies
------------
    pip install pandas numpy scipy matplotlib seaborn

Notes for the hybrid model
--------------------------
* The physics model used here reads from curves/tuned_physics_params.json
  if it exists (output of calibrate_physics_model.py).
  If that file is not found, it falls back to default Tesla Model 3 LR AWD
  specs (Cd=0.23, Crr=0.009) and prints a warning.
* Efficiency (eta_sys) is read directly from the Powertrain_efficiency_gear_SG
  column that already exists in the CSV — no separate curve file needed.
* Temperature is not available in this dataset, so it is not included as a
  feature. If you add a second dataset with ambient temperature, insert it
  as a new column in trip_features.csv and re-run hybrid_model.py.
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION — change these if your files live elsewhere
# ============================================================

PRIMARY_FILE   = "Tracking_data_efficiecny.csv"
PARAMS_FILE    = Path("curves") / "tuned_physics_params.json"   # from calibrate_physics_model.py
OUT_CSV        = Path("data")  / "trip_features.csv"
OUT_FIG_DIR    = Path("figs")

# Segmentation
STOP_THRESHOLD_S   = 30    # seconds of standstill → new trip
MIN_TRIP_DIST_KM   = 0.5   # drop trips shorter than this
MIN_TRIP_DURATION_S = 60   # drop trips shorter than this (seconds)

# Physics constants (defaults — overridden by tuned_physics_params.json if found)
DEFAULT_PARAMS = dict(
    Cd0  = 0.23,
    k_cd = 0.0,
    Crr  = 0.009,
    RHO  = 1.225,
    A    = 2.22,
    MASS = 1847,
    G    = 9.81,
)

HVAC_POWER_W = 0   # AC off during these trips (no temperature data)

# ============================================================
# SETUP
# ============================================================

OUT_CSV.parent.mkdir(exist_ok=True)
OUT_FIG_DIR.mkdir(exist_ok=True)

# ============================================================
# STEP 1 — Load physics params
# ============================================================

if PARAMS_FILE.exists():
    with open(PARAMS_FILE) as fh:
        p = json.load(fh)
    Cd0  = p["Cd0"]
    k_cd = p.get("k_cd", 0.0)
    Crr  = p["Crr"]
    RHO  = p.get("RHO",  DEFAULT_PARAMS["RHO"])
    A    = p.get("A",    DEFAULT_PARAMS["A"])
    MASS = p.get("MASS", DEFAULT_PARAMS["MASS"])
    G    = p.get("G",    DEFAULT_PARAMS["G"])
    print(f"[OK]  Loaded tuned physics params from {PARAMS_FILE}")
    print(f"      Cd0={Cd0:.4f}  Crr={Crr:.4f}  k_cd={k_cd:.6f}")
else:
    print(f"[WARN] {PARAMS_FILE} not found — using default Tesla Model 3 LR AWD specs.")
    print("       Run calibrate_physics_model.py first for best accuracy.")
    Cd0  = DEFAULT_PARAMS["Cd0"]
    k_cd = DEFAULT_PARAMS["k_cd"]
    Crr  = DEFAULT_PARAMS["Crr"]
    RHO  = DEFAULT_PARAMS["RHO"]
    A    = DEFAULT_PARAMS["A"]
    MASS = DEFAULT_PARAMS["MASS"]
    G    = DEFAULT_PARAMS["G"]

# ============================================================
# STEP 2 — Load and clean raw data
# ============================================================

print(f"\n[OK]  Loading {PRIMARY_FILE} ...")
df_raw = pd.read_csv(PRIMARY_FILE)

required = ["Speed", "DeltaT", "Energy Consumption (kWh)",
            "Displacement (m)", "Slope Angle (rad)", "Acceleration",
            "ElevChange", "Powertrain_efficiency_gear_SG"]
missing = [c for c in required if c not in df_raw.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

df = df_raw[required].copy()

# Clip DeltaT: values outside 0–10 s are sensor glitches (56 rows found)
raw_dt = pd.to_numeric(df["DeltaT"], errors="coerce")
n_glitch = ((raw_dt < 0) | (raw_dt > 10)).sum()
df["dt_s"] = raw_dt.clip(0.05, 5.0)

# Speed & geometry
df["v_mps"]     = (pd.to_numeric(df["Speed"],                  errors="coerce").clip(lower=0) / 3.6)
df["theta_rad"] = pd.to_numeric(df["Slope Angle (rad)"],       errors="coerce").fillna(0)
df["accel"]     = pd.to_numeric(df["Acceleration"],            errors="coerce").fillna(0)
df["elev_chg"]  = pd.to_numeric(df["ElevChange"],              errors="coerce").fillna(0)
df["energy"]    = pd.to_numeric(df["Energy Consumption (kWh)"],errors="coerce").fillna(0)
df["disp_m"]    = pd.to_numeric(df["Displacement (m)"],        errors="coerce").fillna(0)

# Powertrain efficiency — clip to physically meaningful range
df["eta_sys"] = pd.to_numeric(df["Powertrain_efficiency_gear_SG"], errors="coerce") \
                  .clip(0.70, 0.98) \
                  .fillna(0.90)   # fallback: mean value

print(f"      Rows loaded      : {len(df_raw):,}")
print(f"      DeltaT glitches  : {n_glitch} rows clipped")
print(f"      Rows after clean : {len(df):,}")

# ============================================================
# STEP 3 — Segment into sub-trips
# ============================================================

# Mark stops (speed < 1 km/h)
df["is_stop"] = df["Speed"].astype(float) < 1.0

# Group consecutive stop/moving stretches
df["stop_group"] = (df["is_stop"] != df["is_stop"].shift()).cumsum()

# Find stop groups whose total duration exceeds threshold
stop_durations = (
    df[df["is_stop"]]
    .groupby("stop_group")["dt_s"]
    .sum()
)
long_stop_groups = set(stop_durations[stop_durations > STOP_THRESHOLD_S].index)

# Trip ID increments at the END of each long stop
df["is_long_stop"] = df["stop_group"].isin(long_stop_groups) & df["is_stop"]
df["trip_id"]      = df["is_long_stop"].cumsum()

n_raw_trips = df["trip_id"].nunique()
print(f"\n[OK]  Segmentation: {len(long_stop_groups)} long stops → {n_raw_trips} raw segments")

# ============================================================
# STEP 4 — Physics model prediction (per row, then aggregated)
# ============================================================

v     = df["v_mps"].to_numpy()
theta = df["theta_rad"].to_numpy()
a     = df["accel"].to_numpy()

Cd_v   = Cd0 + k_cd * v                           # speed-dependent drag (k_cd≈0 if not tuned)
F_aero = 0.5 * RHO * Cd_v * A * v**2
F_rr   = Crr * MASS * G * np.cos(theta)
F_gr   = MASS * G * np.sin(theta)
F_in   = MASS * a
F_tot  = F_aero + F_rr + F_gr + F_in

P_mech_W = F_tot * v + HVAC_POWER_W               # mechanical power (W)

# Direction-aware efficiency
eta = df["eta_sys"].to_numpy()
P_bat_W = np.where(P_mech_W < 0,
                   P_mech_W * eta,     # regen: recover less
                   P_mech_W / eta)     # drive: draw more

df["P_bat_W"]  = P_bat_W
df["E_phys_kWh"] = P_bat_W * df["dt_s"].to_numpy() / 3_600_000   # W·s → kWh

# ============================================================
# STEP 5 — Aggregate to trip-level features
# ============================================================

records = []

for trip_id, seg in df.groupby("trip_id"):
    moving = seg[seg["Speed"].astype(float) >= 1.0]

    dist_km    = seg["disp_m"].sum() / 1000.0
    dur_s      = seg["dt_s"].sum()

    # Drop trips that are too short
    if dist_km < MIN_TRIP_DIST_KM or dur_s < MIN_TRIP_DURATION_S:
        continue

    avg_speed_kph = float(moving["Speed"].mean()) if len(moving) > 0 else 0.0
    max_speed_kph = float(seg["Speed"].max())

    # Elevation
    elev_gain_m   = float(seg["elev_chg"].clip(lower=0).sum())
    elev_loss_m   = float(seg["elev_chg"].clip(upper=0).abs().sum())
    net_elev_m    = float(seg["elev_chg"].sum())

    # Slope
    avg_slope_deg = float(np.degrees(seg["theta_rad"].mean()))
    max_slope_deg = float(np.degrees(seg["theta_rad"].abs().max()))

    # Measured energy
    energy_kWh    = float(seg["energy"].sum())

    # Efficiency (skip if near-zero distance to avoid divide-by-zero)
    efficiency_kWh_100km = (energy_kWh / dist_km * 100) if dist_km > 0.1 else np.nan

    # Regeneration fraction: fraction of time with negative battery power
    regen_rows    = (seg["P_bat_W"] < 0).sum()
    regen_fraction = regen_rows / len(seg) if len(seg) > 0 else 0.0

    # Physics prediction for this sub-trip
    physics_pred_kWh = float(seg["E_phys_kWh"].sum())

    # Residual: what physics got wrong (target for XGBoost)
    residual_kWh  = energy_kWh - physics_pred_kWh

    # Speed tier label (useful for condition-based validation later)
    if avg_speed_kph >= 70:
        speed_tier = "highway"
    elif avg_speed_kph >= 30:
        speed_tier = "mixed"
    else:
        speed_tier = "urban"

    records.append({
        "trip_id"             : int(trip_id),
        "dist_km"             : round(dist_km, 3),
        "duration_min"        : round(dur_s / 60.0, 2),
        "avg_speed_kph"       : round(avg_speed_kph, 2),
        "max_speed_kph"       : round(max_speed_kph, 2),
        "elev_gain_m"         : round(elev_gain_m, 2),
        "elev_loss_m"         : round(elev_loss_m, 2),
        "net_elev_m"          : round(net_elev_m, 2),
        "avg_slope_deg"       : round(avg_slope_deg, 4),
        "max_slope_deg"       : round(max_slope_deg, 2),
        "regen_fraction"      : round(regen_fraction, 4),
        "energy_kWh"          : round(energy_kWh, 5),
        "efficiency_kWh_100km": round(efficiency_kWh_100km, 3) if not np.isnan(efficiency_kWh_100km) else np.nan,
        "physics_pred_kWh"    : round(physics_pred_kWh, 5),
        "residual_kWh"        : round(residual_kWh, 5),
        "speed_tier"          : speed_tier,
    })

trips_df = pd.DataFrame(records)

# ============================================================
# STEP 6 — Summary statistics
# ============================================================

print(f"\n{'='*60}")
print("TRIP DATASET SUMMARY")
print(f"{'='*60}")
print(f"  Valid sub-trips            : {len(trips_df)}")
print(f"  Total distance             : {trips_df['dist_km'].sum():.1f} km")
print(f"  Total measured energy      : {trips_df['energy_kWh'].sum():.3f} kWh")
print(f"  Total physics prediction   : {trips_df['physics_pred_kWh'].sum():.3f} kWh")
total_meas   = trips_df["energy_kWh"].sum()
total_phys   = trips_df["physics_pred_kWh"].sum()
overall_err  = (total_phys - total_meas) / abs(total_meas) * 100
print(f"  Physics overall error      : {overall_err:+.2f}%")
print()

# Per-trip physics MAPE
physics_mape = np.mean(
    np.abs((trips_df["energy_kWh"] - trips_df["physics_pred_kWh"])
           / trips_df["energy_kWh"].replace(0, np.nan))
) * 100
print(f"  Physics per-trip MAPE      : {physics_mape:.2f}%")
print()
print("  Speed tier distribution:")
print(trips_df["speed_tier"].value_counts().to_string())
print()
print("  Feature statistics:")
print(trips_df[["dist_km","avg_speed_kph","elev_gain_m","energy_kWh",
                "efficiency_kWh_100km","residual_kWh"]].describe().round(3).to_string())
print(f"{'='*60}")

# ============================================================
# STEP 7 — Save trip features CSV
# ============================================================

trips_df.to_csv(OUT_CSV, index=False)
print(f"\n[OK]  Saved {len(trips_df)} trips → {OUT_CSV}")

# ============================================================
# STEP 8 — EDA Plots
# ============================================================

PALETTE = {
    "highway" : "#3fb950",
    "mixed"   : "#58a6ff",
    "urban"   : "#e3b341",
}
tier_colors = trips_df["speed_tier"].map(PALETTE)

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor"  : "#161b22",
    "axes.edgecolor"  : "#30363d",
    "axes.labelcolor" : "#c9d1d9",
    "xtick.color"     : "#8b949e",
    "ytick.color"     : "#8b949e",
    "text.color"      : "#e6edf3",
    "grid.color"      : "#21262d",
    "grid.linewidth"  : 0.6,
    "font.family"     : "monospace",
})

# ── Plot 1: Correlation heatmap ──────────────────────────────
num_cols = ["dist_km", "avg_speed_kph", "max_speed_kph",
            "elev_gain_m", "max_slope_deg", "regen_fraction",
            "energy_kWh", "efficiency_kWh_100km",
            "physics_pred_kWh", "residual_kWh"]

corr = trips_df[num_cols].corr()

fig1, ax1 = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            annot=True, fmt=".2f", annot_kws={"size": 8},
            linewidths=0.5, linecolor="#21262d",
            ax=ax1, cbar_kws={"shrink": 0.8})
ax1.set_title("Feature Correlation Matrix — Trip Level", pad=14, fontsize=13)
ax1.tick_params(axis="x", rotation=45)
plt.tight_layout()
fig1.savefig(OUT_FIG_DIR / "eda_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"[OK]  Saved eda_correlation_heatmap.png")

# ── Plot 2: Efficiency vs Average Speed ─────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 5))
valid_mask = trips_df["efficiency_kWh_100km"].notna() & (trips_df["efficiency_kWh_100km"] > 0)
sc2 = ax2.scatter(
    trips_df.loc[valid_mask, "avg_speed_kph"],
    trips_df.loc[valid_mask, "efficiency_kWh_100km"],
    c=tier_colors[valid_mask], s=60, alpha=0.85, edgecolors="#30363d", linewidth=0.5
)
# Draw a smooth trend line
from numpy.polynomial.polynomial import polyfit
x_s = trips_df.loc[valid_mask, "avg_speed_kph"].to_numpy()
y_s = trips_df.loc[valid_mask, "efficiency_kWh_100km"].to_numpy()
sort_idx = np.argsort(x_s)
c2, c1, c0 = polyfit(x_s, y_s, 2)
x_line = np.linspace(x_s.min(), x_s.max(), 200)
ax2.plot(x_line, c0 + c1*x_line + c2*x_line**2,
         color="#f85149", linewidth=1.5, linestyle="--", label="Quadratic trend")
ax2.set_xlabel("Average Speed (km/h)", fontsize=11)
ax2.set_ylabel("Efficiency (kWh / 100 km)", fontsize=11)
ax2.set_title("Energy Efficiency vs Average Speed", fontsize=13, pad=12)
ax2.grid(True, alpha=0.4)
# Legend for speed tiers
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=PALETTE[t], label=t.capitalize()) for t in PALETTE]
legend_els.append(plt.Line2D([0],[0], color="#f85149", linestyle="--", label="Quadratic trend"))
ax2.legend(handles=legend_els, fontsize=9, framealpha=0.3)
plt.tight_layout()
fig2.savefig(OUT_FIG_DIR / "eda_efficiency_vs_speed.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"[OK]  Saved eda_efficiency_vs_speed.png")

# ── Plot 3: Efficiency vs Elevation Gain ────────────────────
fig3, ax3 = plt.subplots(figsize=(9, 5))
ax3.scatter(
    trips_df.loc[valid_mask, "elev_gain_m"],
    trips_df.loc[valid_mask, "efficiency_kWh_100km"],
    c=tier_colors[valid_mask], s=60, alpha=0.85, edgecolors="#30363d", linewidth=0.5
)
ax3.set_xlabel("Elevation Gain (m)", fontsize=11)
ax3.set_ylabel("Efficiency (kWh / 100 km)", fontsize=11)
ax3.set_title("Energy Efficiency vs Elevation Gain", fontsize=13, pad=12)
ax3.grid(True, alpha=0.4)
legend_els2 = [Patch(facecolor=PALETTE[t], label=t.capitalize()) for t in PALETTE]
ax3.legend(handles=legend_els2, fontsize=9, framealpha=0.3)
plt.tight_layout()
fig3.savefig(OUT_FIG_DIR / "eda_efficiency_vs_elevation.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"[OK]  Saved eda_efficiency_vs_elevation.png")

# ── Plot 4: Residual distribution ───────────────────────────
fig4, axes4 = plt.subplots(1, 2, figsize=(11, 5))

ax_hist = axes4[0]
residuals = trips_df["residual_kWh"].dropna()
ax_hist.hist(residuals, bins=20, color="#58a6ff", alpha=0.8, edgecolor="#0d2342")
ax_hist.axvline(0,                   color="#3fb950", linewidth=1.5, linestyle="--", label="Zero")
ax_hist.axvline(residuals.mean(),    color="#e3b341", linewidth=1.5, linestyle=":",  label=f"Mean={residuals.mean():.3f}")
ax_hist.axvline(residuals.median(),  color="#f85149", linewidth=1.5, linestyle="-.", label=f"Median={residuals.median():.3f}")
ax_hist.set_xlabel("Residual (kWh):  measured − physics", fontsize=10)
ax_hist.set_ylabel("Count", fontsize=10)
ax_hist.set_title("Physics Model Residuals", fontsize=12)
ax_hist.legend(fontsize=8, framealpha=0.3)
ax_hist.grid(True, alpha=0.4)

ax_sc = axes4[1]
ax_sc.scatter(trips_df["physics_pred_kWh"], trips_df["energy_kWh"],
              c=tier_colors, s=60, alpha=0.85, edgecolors="#30363d", linewidth=0.5)
lims = [min(trips_df["energy_kWh"].min(), trips_df["physics_pred_kWh"].min()) - 0.5,
        max(trips_df["energy_kWh"].max(), trips_df["physics_pred_kWh"].max()) + 0.5]
ax_sc.plot(lims, lims, color="#3fb950", linewidth=1.2, linestyle="--", label="Perfect prediction")
ax_sc.set_xlim(lims); ax_sc.set_ylim(lims)
ax_sc.set_xlabel("Physics Prediction (kWh)", fontsize=10)
ax_sc.set_ylabel("Measured Energy (kWh)",    fontsize=10)
ax_sc.set_title("Physics vs Measured — Scatter", fontsize=12)
ax_sc.legend(fontsize=8, framealpha=0.3)
ax_sc.grid(True, alpha=0.4)

plt.suptitle("Physics Model Residual Analysis", fontsize=13, y=1.01)
plt.tight_layout()
fig4.savefig(OUT_FIG_DIR / "eda_residuals.png", dpi=150, bbox_inches="tight")
plt.close(fig4)
print(f"[OK]  Saved eda_residuals.png")

# ============================================================
# DONE
# ============================================================

print(f"""
{'='*60}
DONE — outputs written:
  {OUT_CSV}
  {OUT_FIG_DIR}/eda_correlation_heatmap.png
  {OUT_FIG_DIR}/eda_efficiency_vs_speed.png
  {OUT_FIG_DIR}/eda_efficiency_vs_elevation.png
  {OUT_FIG_DIR}/eda_residuals.png

NEXT STEP:
  Run  hybrid_model.py  to fit the XGBoost correction layer
  on the residual column in  {OUT_CSV}
{'='*60}
""")
