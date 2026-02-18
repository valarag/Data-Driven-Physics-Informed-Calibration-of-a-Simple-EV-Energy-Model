import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Tracking_data_efficiecny.csv")

# Keep only what we care about
df = df[[
    "Speed",
    "Motor_rpm_gear_SG",
    "Motor_torque_gear_SG",
    "Powertrain_efficiency_gear_SG"
]]

print("\n=== BASIC INFO ===")
print(df.describe())

# 1) Efficiency distribution
plt.figure()
plt.hist(df["Powertrain_efficiency_gear_SG"], bins=100)
plt.title("Powertrain Efficiency Distribution")
plt.xlabel("Efficiency")
plt.ylabel("Count")
plt.show()

# 2) Efficiency vs Speed
plt.figure()
plt.scatter(df["Speed"], df["Powertrain_efficiency_gear_SG"], s=2)
plt.title("Efficiency vs Speed")
plt.xlabel("Speed (km/h)")
plt.ylabel("Efficiency")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load ONE file manually
df = pd.read_csv("2_preprocessed/JY_SOC_25deg/FTP-75.csv")

print("\n=== BASIC INFO ===")
print(df.describe())

# Power
df["Power_kW"] = (df["Voltage"] * df["Current"]) / 1000

# 1) Current distribution
plt.figure()
plt.hist(df["Current"], bins=100)
plt.title("Current Distribution")
plt.show()

# 2) Voltage vs Time
plt.figure()
plt.plot(df["Time"], df["Voltage"])
plt.title("Voltage over Time")
plt.show()

# 3) Power vs Time
plt.figure()
plt.plot(df["Time"], df["Power_kW"])
plt.title("Power over Time")
plt.show()
