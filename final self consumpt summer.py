'''
TThis code represents the equations for calculating PV production based on temperature correction model and Irradiance
data from SAM as deccriped in the methodology.
It also has the load input in order to estimate PV surpluses for the various scenarios (I,II,III) for the winter,
summer and yearly

The results is sent to the grid model in panadpoweer
It also rpresents the summaeruy of the variosu configurations.
'''





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# 1) LOAD DATA from winter house load profile,summer houseload profile and yearly load nedu profile
###############################################################################
csv_path = r"C:/Users/cusa/Downloads/data studiesnedu.csv" #change file based on the season or yearly studies
#This file is for the swinter containing actual data of the 9 participants. temperature and irradiance during the studie period
df = pd.read_csv(csv_path)

print("Columns:", df.columns.tolist())
print(df.head())

# Columns: Hour, Load_W_XXXX, Temperature_C, Irradiance_Wm2...
# Keep only 168 rows (one week) for either summer or winter simulaiton or  a yearis 8760
n = 8760
df = df.iloc[:n]


###############################################################################
# 2) DEFINE HOUSES, PV, BATTERY, PEAK HOURS
###############################################################################
house_ids = [2284, 4243, 4244, 5132, 7683, 7690, 8411, 8617, 9302]

pv_capacities = {
    "2284": 4.2,
    "4243": 25,
    "4244": 5.2,
    "5132": 8,
    "7683": 10,
    "7690": 6.75,
    "8411": 3.68,
    "8617": 9,
    "9302": 11
}

battery_power_kW_each     =15
battery_capacity_kWh_each = 10.5
num_houses = len(house_ids)

# Central battery
central_battery_power_kW      = battery_power_kW_each     * num_houses
central_battery_capacity_kWh  = battery_capacity_kWh_each * num_houses

# Central PV
central_pv_kW = sum(pv_capacities.values())  # ~82.83 kW

# Example: define peak hours for e.g. 5pm–8pm
peak_hours_set = {17, 18, 19, 20}

# Extract irradiance (W/m²) and temperature (°C) arrays
irr = df["Irradiance_Wm2"].values
ambient_temp = df["Temperature_C"].values


###############################################################################
# 3) TEMPERATURE-AWARE PV CALCULATION
###############################################################################
def get_pv_power_temp(pv_kw, irr_wm2, ambient_temp_c):
    """
    Simple NOCT-based model plus linear temperature coefficient (typical).
    """
    stc_irr = 1000.0
    stc_temp= 25.0
    noct_irr= 800.0
    noct_temp=45.0
    temp_coeff= -0.004  # -0.4%/°C

    cell_temp = ambient_temp_c + (irr_wm2/noct_irr)*(noct_temp-20.0)
    temp_factor= 1.0 + temp_coeff*(cell_temp - stc_temp)
    irr_ratio  = irr_wm2/stc_irr

    power_kW= pv_kw * irr_ratio * temp_factor
    return max(power_kW, 0.0)


###############################################################################
# 4) BASELINE (NO BATTERY)
###############################################################################
def baseline_no_battery(load_kW, pv_kW):
    """
    Return arrays: grid_import, pv_export, net_load, peak_import.
    net_load>0 => import, net_load<0 => export.
    """
    n = len(load_kW)
    grid_import = np.zeros(n)
    pv_export   = np.zeros(n)
    net_load    = np.zeros(n)

    for i in range(n):
        if load_kW[i] > pv_kW[i]:
            # leftover load => grid
            grid_import[i] = load_kW[i]-pv_kW[i]
            net_load[i]    = grid_import[i]
        else:
            # leftover PV => export
            pv_export[i] = pv_kW[i]-load_kW[i]
            net_load[i]  = -pv_export[i]

    peak_import= grid_import.max()
    return grid_import, pv_export, net_load, peak_import


###############################################################################
# 5) SELF-CONSUMPTION DISPATCH with 95% EFFICIENCY
###############################################################################
def self_consumption_dispatch_eff(
    load_kW, pv_kW,
    battery_power_kW,
    battery_capacity_kWh,
    charge_eff=0.95,
    discharge_eff=0.95,
    init_soc_frac=0.5,
    min_soc_frac=0.2,
    max_soc_frac=0.9,
    allow_grid_charge=False
):
    """
    Self-consumption with battery charge/discharge efficiency ~95%.
     - At each hour:
       * direct_pv_use = min(load, pv)  (no losses)
       * surplus -> battery (some lost in charge)
       * deficit -> battery discharge (some lost in discharge)
    Returns dict with final_net_load, grid_import, pv_export, soc arrays,
    plus total pv_direct_use, pv_to_battery, charge_loss, discharge_loss.
    """
    n = len(load_kW)
    dt = 1.0

    soc = np.zeros(n)
    final_net_load= np.zeros(n)
    grid_import= np.zeros(n)
    pv_export=  np.zeros(n)

    # Additional trackers
    pv_direct_use = 0.0
    pv_to_battery = 0.0
    charge_loss   = 0.0
    discharge_loss= 0.0

    # Battery SoC constraints
    min_soc = min_soc_frac * battery_capacity_kWh
    max_soc = max_soc_frac * battery_capacity_kWh
    soc_current= init_soc_frac * battery_capacity_kWh

    for i in range(n):
        L = load_kW[i]
        P = pv_kW[i]

        # 1) Direct PV usage
        direct_use= min(L, P)
        pv_direct_use += direct_use

        remain_load   = L - direct_use
        surplus_pv    = P - direct_use

        # 2) Surplus => charge battery
        if surplus_pv>0:
            possible_charge_kW= min(battery_power_kW, surplus_pv)
            possible_charge_kWh= possible_charge_kW*dt
            avail_capacity= max_soc - soc_current
            if avail_capacity<0: avail_capacity=0
            actual_charge_kWh= min(possible_charge_kWh, avail_capacity)

            # battery sees 'actual_charge_kWh * charge_eff' inside
            # difference is lost
            battery_input= actual_charge_kWh*charge_eff
            this_charge_loss= actual_charge_kWh- battery_input
            charge_loss+= this_charge_loss
            pv_to_battery+= actual_charge_kWh

            soc_current+= battery_input
            surplus_pv-= actual_charge_kWh

            # leftover surplus => export
            if surplus_pv>0:
                pv_export[i]= surplus_pv
                final_net_load[i]= -surplus_pv
            else:
                final_net_load[i]= 0.0

        # 3) If remain_load>0 => discharge battery
        if remain_load>0:
            available_discharge= soc_current- min_soc
            if available_discharge<0:
                available_discharge=0
            possible_discharge_kW= min(battery_power_kW, remain_load)
            possible_discharge_kWh= possible_discharge_kW*dt
            actual_discharge_kWh= min(possible_discharge_kWh, available_discharge)

            battery_output= actual_discharge_kWh*discharge_eff
            this_dis_loss= actual_discharge_kWh- battery_output
            discharge_loss+= this_dis_loss

            soc_current-= actual_discharge_kWh

            remain_load-= battery_output

            # leftover => grid
            if remain_load>0:
                grid_import[i]= remain_load
                final_net_load[i]= remain_load
            else:
                final_net_load[i]=0.0

        # 4) (Optionally allow grid charge)The battery is lkely charged from the grid in winter or low Irr days
        if allow_grid_charge:
            pass

        # Clip SoC
        soc_current= max(min_soc, min(soc_current, max_soc))
        soc[i]= soc_current

    # Summaries representing no specific cases and serving as the baseline.
    peak_after_batt= grid_import.max()
    return {
        "final_net_load": final_net_load,
        "grid_import":    grid_import,
        "pv_export":      pv_export,
        "soc":            soc,

        "pv_direct_use":  pv_direct_use,
        "pv_to_battery":  pv_to_battery,
        "charge_loss":    charge_loss,
        "discharge_loss": discharge_loss,
        "peak_after_batt":peak_after_batt
    }


###############################################################################
# 6) PEAK-HOUR HELPER
###############################################################################
def sum_in_peak_hours(net_array, peak_hours):
    total=0.0
    n= len(net_array)
    for i in range(n):
        if (i%24) in peak_hours and net_array[i]>0:
            total+= net_array[i]
    return total


###############################################################################
# 7) SCENARIO IMPLEMENTATIONS (with SoC% tracking)
###############################################################################
def scenario_i_each_house(df, irr, temp, peak_hours):
    """
    (i) Each house has own battery + PV + 95% eff. SoC in % is average across houses.
    """
    n= len(irr)
    baseline_net= np.zeros(n)
    final_net=    np.zeros(n)
    grid_import=  np.zeros(n)
    pv_export=    np.zeros(n)

    total_pv_prod= 0.0
    total_pv_direct= 0.0
    total_pv_store= 0.0
    total_charge_loss=0.0
    total_dis_loss= 0.0

    soc_arrays= []

    for hid in house_ids:
        load_kW= df[f"Load_W_{hid}"].values[:n]
        pv_kW=   np.array([get_pv_power_temp(pv_capacities[str(hid)], irr[i], temp[i]) for i in range(n)])
        total_pv_prod+= pv_kW.sum()

        # baseline
        b_grid, b_exp, b_net, b_peak= baseline_no_battery(load_kW, pv_kW)
        baseline_net+= b_net

        # battery
        res= self_consumption_dispatch_eff(load_kW, pv_kW,
                                           battery_power_kW_each,
                                           battery_capacity_kWh_each,
                                           charge_eff=0.95, discharge_eff=0.95)
        final_net   += res["final_net_load"]
        grid_import += res["grid_import"]
        pv_export   += res["pv_export"]
        total_pv_direct+= res["pv_direct_use"]
        total_pv_store += res["pv_to_battery"]
        total_charge_loss+= res["charge_loss"]
        total_dis_loss   += res["discharge_loss"]

        soc_arrays.append(res["soc"])  # in kWh

    # Convert average SoC in kWh to SoC% (house battery each)
    avg_soc_kWh= np.mean(soc_arrays, axis=0)  # hour by hour
    soc_percent= (avg_soc_kWh / battery_capacity_kWh_each)*100.0

    # Peak-hour sums
    peak_sum_no_batt= sum_in_peak_hours(baseline_net, peak_hours)
    peak_sum_with_batt= sum_in_peak_hours(final_net, peak_hours)
    peak_hour_reduction= peak_sum_no_batt- peak_sum_with_batt
    peak_red_pct= (peak_hour_reduction/peak_sum_no_batt*100.0) if peak_sum_no_batt>1e-9 else 0.0

    # Single-hour peaks
    base_peak= baseline_net[baseline_net>0].max() if (baseline_net>0).any() else 0.0
    final_peak= final_net[final_net>0].max() if (final_net>0).any() else 0.0

    total_import= grid_import.sum()
    total_export= pv_export.sum()
    print(pv_export)
    sc_ratio= 1.0
    if total_pv_prod>1e-9:
        sc_ratio= 1.0- (total_export / total_pv_prod)

    return {
        "baseline_net_load": baseline_net,
        "final_net_load":    final_net,
        "grid_import_kWh":   total_import,
        "pv_export_kWh":     total_export,
        "peak_sum_no_batt_kWh": peak_sum_no_batt,
        "peak_sum_with_batt_kWh": peak_sum_with_batt,
        "peak_hour_reduction_kWh": peak_hour_reduction,
        "peak_hour_reduction_pct": peak_red_pct,
        "baseline_single_hour_peak": base_peak,
        "final_single_hour_peak":    final_peak,
        "self_consumption": sc_ratio,
         "pv_export":pv_export,
        # Additional
        "total_pv_production": total_pv_prod,
        "total_pv_direct_use": total_pv_direct,
        "total_pv_stored":     total_pv_store,
        "charge_loss":         total_charge_loss,
        "discharge_loss":      total_dis_loss,

        # SoC% array for plotting
        "soc_percent_array": soc_percent,

        # For plotting
        "imports_array": grid_import,
        "exports_array": pv_export,
    }


def scenario_ii_central_battery(df, irr, temp, peak_hours):
    """
    (ii) One central battery + distributed PV (95% eff).
    SoC% is a single battery => we directly convert 'res["soc"]' to percent.
    """
    n= len(irr)
    total_load= np.zeros(n)
    total_pv=   np.zeros(n)

    for hid in house_ids:
        load_kW= df[f"Load_W_{hid}"].values[:n]
        pv_kW= np.array([get_pv_power_temp(pv_capacities[str(hid)], irr[i], temp[i]) for i in range(n)])
        total_load+= load_kW
        total_pv  += pv_kW

    b_grid, b_exp, b_net, b_peak= baseline_no_battery(total_load, total_pv)

    res= self_consumption_dispatch_eff(total_load, total_pv,
                                       central_battery_power_kW,
                                       central_battery_capacity_kWh,
                                       charge_eff=0.95, discharge_eff=0.95)

    final_net= res["final_net_load"]
    peak_sum_no_batt= sum_in_peak_hours(b_net, peak_hours)
    peak_sum_with_batt= sum_in_peak_hours(final_net, peak_hours)
    peak_hour_reduction= peak_sum_no_batt- peak_sum_with_batt
    peak_red_pct= (peak_hour_reduction/peak_sum_no_batt*100) if peak_sum_no_batt>1e-9 else 0.0

    base_peak= b_net[b_net>0].max() if (b_net>0).any() else 0.0
    fin_peak= final_net[final_net>0].max() if (final_net>0).any() else 0.0

    total_import= res["grid_import"].sum()
    total_export= res["pv_export"].sum()
    total_pv_prod= total_pv.sum()
    sc_ratio= 1.0
    if total_pv_prod>1e-9:
        sc_ratio= 1.0- (total_export/ total_pv_prod)

    # Convert battery SoC from kWh to % across time
    soc_kWh= res["soc"]  # array of length n
    soc_percent= (soc_kWh / central_battery_capacity_kWh)*100.0

    return {
        "baseline_net_load": b_net,
        "final_net_load":    final_net,
        "grid_import_kWh":   total_import,
        "pv_export_kWh":     total_export,
        "peak_sum_no_batt_kWh": peak_sum_no_batt,
        "peak_sum_with_batt_kWh": peak_sum_with_batt,
        "peak_hour_reduction_kWh": peak_hour_reduction,
        "peak_hour_reduction_pct": peak_red_pct,
        "baseline_single_hour_peak": base_peak,
        "final_single_hour_peak":    fin_peak,
        "self_consumption": sc_ratio,

        "total_pv_production": total_pv_prod,
        "total_pv_direct_use": res["pv_direct_use"],
        "total_pv_stored":     res["pv_to_battery"],
        "charge_loss":         res["charge_loss"],
        "discharge_loss":      res["discharge_loss"],

        "soc_percent_array":   soc_percent,

        "imports_array": res["grid_import"],
        "exports_array": res["pv_export"],
    }


def scenario_iii_central_pv(df, irr, temp, peak_hours):
    """
    (iii) One central PV farm + each house battery => 95% eff.
    We'll average the SoC% across houses (like scenario i).
    """
    n= len(irr)
    pv_central= np.array([get_pv_power_temp(central_pv_kW, irr[i], temp[i]) for i in range(n)])
    total_load= np.zeros(n)

    for hid in house_ids:
        total_load+= df[f"Load_W_{hid}"].values[:n]

    base_net_agg= np.zeros(n)
    final_net_agg= np.zeros(n)
    grid_imp_agg=  np.zeros(n)
    pv_exp_agg=    np.zeros(n)

    total_pv_direct=0.0
    total_pv_store=0.0
    total_charge_loss=0.0
    total_dis_loss= 0.0

    soc_arrays= []

    epsilon=1e-9

    for hid in house_ids:
        load_kw= df[f"Load_W_{hid}"].values[:n]
        fraction= np.where(total_load<epsilon, 0.0, load_kw/ total_load)
        house_pv= pv_central * fraction

        b_grid, b_exp, b_net, b_peak= baseline_no_battery(load_kw, house_pv)
        base_net_agg+= b_net

        res= self_consumption_dispatch_eff(load_kw, house_pv,
                                           battery_power_kW_each, battery_capacity_kWh_each,
                                           charge_eff=0.95, discharge_eff=0.95)
        final_net_agg+= res["final_net_load"]
        grid_imp_agg += res["grid_import"]
        pv_exp_agg   += res["pv_export"]

        total_pv_direct+= res["pv_direct_use"]
        total_pv_store += res["pv_to_battery"]
        total_charge_loss+= res["charge_loss"]
        total_dis_loss   += res["discharge_loss"]

        soc_arrays.append(res["soc"])

    # Average SoC across all houses => convert to % of each house's capacity
    avg_soc_kWh= np.mean(soc_arrays, axis=0)
    soc_percent= (avg_soc_kWh / battery_capacity_kWh_each)*100.0

    peak_sum_no_batt= sum_in_peak_hours(base_net_agg, peak_hours)
    peak_sum_with_batt= sum_in_peak_hours(final_net_agg, peak_hours)
    peak_hour_reduction= peak_sum_no_batt- peak_sum_with_batt
    peak_red_pct= (peak_hour_reduction/peak_sum_no_batt*100) if peak_sum_no_batt>1e-9 else 0.0

    base_peak= base_net_agg[base_net_agg>0].max() if (base_net_agg>0).any() else 0.0
    fin_peak= final_net_agg[final_net_agg>0].max() if (final_net_agg>0).any() else 0.0

    total_import= grid_imp_agg.sum()
    total_export= pv_exp_agg.sum()
    total_pv_prod= pv_central.sum()
    sc_ratio= 1.0
    if total_pv_prod>1e-9:
        sc_ratio= 1.0- (total_export/ total_pv_prod)

    return {
        "baseline_net_load": base_net_agg,
        "final_net_load":    final_net_agg,
        "grid_import_kWh":   total_import,
        "pv_export_kWh":     total_export,
        "peak_sum_no_batt_kWh": peak_sum_no_batt,
        "peak_sum_with_batt_kWh": peak_sum_with_batt,
        "peak_hour_reduction_kWh": peak_hour_reduction,
        "peak_hour_reduction_pct": peak_red_pct,
        "baseline_single_hour_peak": base_peak,
        "final_single_hour_peak":    fin_peak,
        "self_consumption": sc_ratio,

        "total_pv_production": total_pv_prod,
        "total_pv_direct_use": total_pv_direct,
        "total_pv_stored":     total_pv_store,
        "charge_loss":         total_charge_loss,
        "discharge_loss":      total_dis_loss,

        "soc_percent_array":   soc_percent,

        "imports_array": grid_imp_agg,
        "exports_array": pv_exp_agg,
    }


###############################################################################
# 8) RUN SCENARIOS
###############################################################################
res_i   = scenario_i_each_house(df, irr, ambient_temp, peak_hours_set)
res_ii  = scenario_ii_central_battery(df, irr, ambient_temp, peak_hours_set)
res_iii = scenario_iii_central_pv(df, irr, ambient_temp, peak_hours_set)

###############################################################################
# 9) PRINT RESULTS
###############################################################################
def print_scenario_results(name, r):
    print(f"\n=== {name} ===")
    print(f" Peak-Hour Summation (No Batt): {r['peak_sum_no_batt_kWh']:.2f} kWh")
    print(f" Peak-Hour Summation (WithBatt):{r['peak_sum_with_batt_kWh']:.2f} kWh")
    print(f" Peak-Hour Reduction:          {r['peak_hour_reduction_kWh']:.2f} kWh"
          f" ({r['peak_hour_reduction_pct']:.1f} %)")

    print(f" Baseline Single-Hour Peak: {r['baseline_single_hour_peak']:.2f} kW")
    print(f" Final Single-Hour Peak:    {r['final_single_hour_peak']:.2f} kW")

    print(f" Grid Import (week):       {r['grid_import_kWh']:.2f} kWh")
    print(f" PV Export (week):         {r['pv_export_kWh']:.2f} kWh")
    print(f" Self-Consumption Ratio:   {r['self_consumption']*100:.2f} %")

    print(f" Total PV Production:      {r['total_pv_production']:.2f} kWh")
    print(f" PV Direct Use:            {r['total_pv_direct_use']:.2f} kWh")
    print(f" PV Stored in Battery:     {r['total_pv_stored']:.2f} kWh")

    print(f" Charge Losses:            {r['charge_loss']:.2f} kWh")
    print(f" Discharge Losses:         {r['discharge_loss']:.2f} kWh")

print_scenario_results("(i) Each House PV+Battery (95% eff)",     res_i)
print_scenario_results("(ii) Central Battery, Dist. PV (95% eff)",res_ii)
print_scenario_results("(iii) Central PV + Dist. Battery(95% eff)",res_iii)


###############################################################################
# 10) PLOTS
###############################################################################
hours = np.arange(n)

font = { 'size': 20}  
plt.rcParams['axes.labelweight'] = 'normal'        
plt.rc('font', **font)
# (a) Grid Import & PV Export
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

ax[0].plot(hours, res_i["imports_array"], label="Grid Import",  color="tab:blue")
ax[0].plot(hours, res_i["exports_array"], label="PV Export",    color="tab:orange")
ax[0].set_ylabel("kW",fontsize=18)
#ax[0].set_title("(i) Each House (95% eff) - Grid Import & PV Export")
ax[0].legend(loc="best",fontsize=12)

ax[1].plot(hours, res_ii["imports_array"], label="Grid Import", color="tab:blue")
ax[1].plot(hours, res_ii["exports_array"], label="PV Export",   color="tab:orange")
ax[1].set_ylabel("kW",fontsize=18)
ax[1].set_title("(ii) Central Battery (95% eff) - Grid Import & PV Export")
ax[1].legend(loc="best",fontsize=12)

ax[2].plot(hours, res_iii["imports_array"], label="Grid Import", color="tab:blue")
ax[2].plot(hours, res_iii["exports_array"], label="PV Export",   color="tab:orange")
ax[2].set_xlabel("Hour over a week",fontsize=18)
ax[2].set_ylabel("kW")
ax[2].set_title("(iii) Central PV + Dist. Battery (95% eff) - Grid Import & Export")
ax[2].legend(loc="upper right",fontsize=12)

plt.tight_layout()
plt.show()


# (b) SoC in Percent
fig2, ax2 = plt.subplots(figsize=(10, 8))

ax2.plot(hours, res_i["soc_percent_array"],   label="(I) House SoC% (avg)", alpha=0.8)
ax2.plot(hours, res_ii["soc_percent_array"],  label="(II) Central Battery SoC%", alpha=0.8)
ax2.plot(hours, res_iii["soc_percent_array"], label="(III) House SoC% (avg)", alpha=0.8)

ax2.set_xlabel("Hour")
ax2.set_ylabel("State of Charge (%)")
#ax2.set_title("Battery SoC in % (95% Efficiency)")
ax2.legend(loc="best",fontsize=12)


plt.tight_layout()
plt.show()


# (c) Net Load (baseline vs battery)
fig3, ax3 = plt.subplots(3,1, figsize=(10,8), sharex=True)

ax3[0].plot(hours, res_i["baseline_net_load"], label="I Baseline Net Load", color="gray", alpha=0.7)
ax3[0].plot(hours, res_i["final_net_load"],    label="After Battery",     color="black")
ax3[0].axhline(0, color='red', linestyle='--')
ax3[0].set_ylabel("kW",fontsize=18)

#ax3[0].set_title("(i) Net Load w/ & w/o Battery (95% eff)")
ax3[0].legend(loc="lower right",fontsize=12)

ax3[1].plot(hours, res_ii["baseline_net_load"], label="II Baseline Net Load", color="gray", alpha=0.7)
ax3[1].plot(hours, res_ii["final_net_load"],    label="After Battery",     color="black")
ax3[1].axhline(0, color='red', linestyle='--')
ax3[1].set_ylabel("kW",fontsize=18)
#ax3[1].set_title("(ii) Net Load w/ & w/o Battery (95% eff)")
ax3[1].legend(loc="lower right",fontsize=12)

ax3[2].plot(hours, res_iii["baseline_net_load"], label="III Baseline Net Load", color="gray", alpha=0.7)
ax3[2].plot(hours, res_iii["final_net_load"],    label="After Battery",     color="black")
ax3[2].axhline(0, color='red', linestyle='--')
ax3[2].set_xlabel("Hour")
ax3[2].set_ylabel("kW",fontsize=18)
#ax3[2].set_title("(iii) Net Load w/ & w/o Battery (95% eff)")
ax3[2].legend(loc="lower right",fontsize=12)


