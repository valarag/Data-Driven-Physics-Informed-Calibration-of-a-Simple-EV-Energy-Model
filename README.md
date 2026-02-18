# Lithium-Ion Battery Drive Cycle Dataset
This dataset was developed by Jiaqi Yao at Technische Universität Berlin (jiaqi.yao97@gmail.com / jiaqi.yao@tu-berlin.de). Please offer appropriate citations if you would like to use the data.

## General Info
Three brand-new LG INR 21700 M50LT lithium-ion battery cells with 4.93Ah nominal capacity under the same aging state were cycled with 12 drive cycles (4 cycles each cell) under 5 ambient temperatures (45°C, 35°C, 25°C, 15°C, 5°C). The 12 cycles are: Braunschweig City Driving Cycle (BCDC), California Unified Cycle (LA92), Heavy Heavy-Duty Diesel Truck Composite Cycle (HHDDT), City Suburban Heavy Vehicle Cycle (CSHVC), Federal Test Procedure-72 (FTP-72), Federal Test Procedure-75 (FTP-75), Highway Fuel Economy Test (HWFET), Inspection and Maintenance (IM), US06 Supplemental Federal Test Procedure (US06), Parcel Delivery Truck Cycle Baltimore (PDTCB), Port Drayage Metro Highway Cycle California (PDMHC), Orange County Transit Bus Cycle (OCTBC).

The exact assignment of the drive cycles is listed in the following table. Note that the order of the cycles in the table for each cell corresponds to the order in the test plan.

| Cell Name | Drive Cycles                         |
| --------- | ------------------------------------ |
| NMC063    | 1_BCDC, 2_LA92, 3_HHDDT, 8_IM        |
| NMC068    | 4_CSHVC, 5_FTP-72, 6_FTP-75, 7_HWFET |
| NMC069    | 9_US06, 10_PDTCB, 11_PDMHC, 12_OCTBC |

## Test Plan
Before each test started, the ambient temperature was set using the Memmert climate chamber. The test was started after the reading of the temperature sensor is stable. The test plan of each cell at each temperature is as follows:
1. Relaxation for 1min (1Hz)
2. CC charging with 1C until the cell voltage reaches 4.2V (10Hz)
3. CV charging with 4.2V until the current drops under 0.02C (10Hz)
4. Relaxation for 2h (1Hz)
5. Repeated cycling with one drive cycle until the cell voltage reaches 2.5V (10Hz)
6. Repeat step 2-5 until the assigned four drive cycles are completed
Note that the regeneration at the beginning of each dynamic cycle (Above 90% SOC pre-calculated using the power profile and nominal voltage) is trimmed to prevent overcharge. Regeneration power for the low-temperature case (5°C) is multiplied with a factor of 0.7 to simulate reality.

## Files
There are in total three independent folders given:

| Folder Name    | Description                                                                                                                                                                                         |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0_raw          | Raw test data (including capacity test at 25°C and complete test procedures of all cells)                                                                                                           |
| 1_split        | Drive cycle data extracted from raw test data and split into individual profiles with reference SOC calculated using coulomb counting, each starting from 100% SOC to end-of-discharge voltage 2.5V |
| 2_preprocessed | Drive cycle data from 1_split resampled into strict 10Hz  (plug and play!)                                                                                                                          |
The naming convention is very intuitive. For example, in root folder "2_preprocessed" you can find a folder named "JY_SOC_5deg", which means this folder contains data under 5°C. And in this folder, you can further find 12 .csv files with the 12 preprocessed drive cycle profiles under 5°C. The names of the drive cycles are explicitly stated in the file name.

## Data
In the raw test data from folder "0_raw", the files were directly exported from the BaSyTec CTS cycler. The data contains standard test records with units explicitly stated. In case of any confusion, you can refer to the documentation.
In the data from folder "1_split" and "2_preprocessed", the following columns are used:
Time: time elapsed from the beginning of the drive cycle in second
Voltage: measured terminal voltage of the cell in volt
Current: measured current in ampere
Temperature: measured cell surface temperature in degree Celsius
Power: measured power in watt, not included in preprocessed data
SOC: reference state of charge of the cell calculated using coulomb counting
