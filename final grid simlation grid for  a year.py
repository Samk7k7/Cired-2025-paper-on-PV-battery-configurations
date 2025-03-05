

#****************************************************************************
'''
This code analysis the effects of central battery, single individual batteries with PV mapping and central
PV with individual battery mapping on the grid

The results present the line and transformer laoding, voltage deviations of each configuration.

The final results are summarised in the CIRED paper presented by Samuel Cudjoe on the topic

INTEGRATED PV-BATTERY DESIGN MAPPING STRATEGIES AND CONTROL IN 
DECENTRALISED ENERGY COMMUNITIES FOR ENHANCED GRID CONGESTION RELIEF
Samuel Cudjoe1*
, Marten van der Laan2
, Koen Kok3
'''
#****************************************************************************

#100%|██████████| 8760/8760 [01:33<00:00, 124.61it/s] simulation time for a year at hourly time step

import numpy as np
import pandas as pd

import pandapower as pp
import pandapower.control as control
import pandapower.timeseries as timeseries
from pandapower.timeseries.data_sources.frame_data import DFData
import pandapower.plotting as plot
import pickle
import matplotlib.pyplot as plt
import os

# ****************************************************************************
# 1) Load the pandapower network : In this work, the Ansen real grid was used
# Similar test grid can be created here in pandapower or imported from existing grid.
# ****************************************************************************
file_path_network = r"C:/Users/cusa/Downloads/Grid data/Grid data/Ansen_OM DE KAMP 14_network.pkl"
with open(file_path_network, "rb") as f:
    data = pickle.load(f)
    net = data["network"]

# Create an external grid at the trafo HV bus (if not already present)
for index, row in net.trafo.iterrows():
    pp.create_ext_grid(net, bus=row.hv_bus, vm_pu=1.05, name="Grid Connection")

# ****************************************************************************
# 2) Define the buses that contain loads (given in your question)
# ****************************************************************************
'''
buses_with_load = [
    334, 335, 349, 350, 351, 352, 353, 354, 355, 356
    ]
 ''' 
    
   


buses_with_load=[350,359,386,390,397,429,446,376,375]

#****************************************************************************
# 3) Read the time series CSV & create new columns for "divided by 9" shares
#This time series was the results of the energy flows from the various configurations(three different mappings)
#It was analysed using the previous energy flow codes and validated with SAM as explained in the methodology

# ****************************************************************************
#file_path_timeseries = r"C:/Users/cusa/Downloads/voltage study data setb.csv"
file_path_timeseries = r"C:/Users/cusa/Downloads/nedupower.csv"
df_ts = pd.read_csv(file_path_timeseries)

# Basic checks to avoid missing  data
n_ts = len(df_ts["time_step"])
print("Number of time steps:", n_ts)
print("Time series data columns:", df_ts.columns)


#From the results of the energy flow simulations, positive is referred as load and negative as sgen(PV generation)
df_ts["B_load_segment"] = df_ts["SY3L"] / 9
df_ts["B_gen_segment"]  = df_ts["SY3_G"] / 9

# Create a DataSource from the DataFrame
ds = DFData(df_ts)

# ****************************************************************************
# 4) For each bus in the list, create a load + sgen and link them to the new columns
# ****************************************************************************
for bus_id in buses_with_load:
    # Create a load (initial p_mw=0; we'll control it via ConstControl)
    load_idx = pp.create_load(net, bus=bus_id, p_mw=1, q_mvar=0.0,
                              name=f"Load_Bus{bus_id}")
    # Create a solar PV sgen at the same bus
    sgen_idx = pp.create_sgen(net, bus=bus_id, p_mw=1, q_mvar=0.0,
                              name=f"Sgen_Bus{bus_id}", type="pv")

    # Tie the load to "B_load_segment"
    control.ConstControl(
        net,
        element='load',
        element_index=load_idx,
        variable='p_mw',
        data_source=ds,
        profile_name="B_load_segment"
    )

    # Tie the sgen to "B_gen_segment"
    control.ConstControl(
        net,
        element='sgen',
        element_index=sgen_idx,
        variable='p_mw',
        data_source=ds,
        profile_name="B_gen_segment"
    )

# ****************************************************************************
# 5) Setup an OutputWriter to store results (Excel)
# ****************************************************************************
output_dir = "./trial grid/"
os.makedirs(output_dir, exist_ok=True)

ow = timeseries.OutputWriter(
    net,
    time_steps=range(n_ts),
    output_path=output_dir,
    output_file_type=".xlsx"
)

# Log bus voltages, line loading, and trafo loading
ow.log_variable('res_bus', 'vm_pu')
ow.log_variable('res_line', 'loading_percent')
ow.log_variable('res_trafo', 'loading_percent')

# ****************************************************************************
# 6) Run the time series simulation
# ****************************************************************************
timeseries.run_timeseries(net, time_steps=range(n_ts))

# ****************************************************************************
# 7) Read the logged Excel results
# ****************************************************************************
bus_vm_df = pd.read_excel(
    os.path.join(output_dir, "res_bus", "vm_pu.xlsx"),
    index_col=0
)
line_loading_df = pd.read_excel(
    os.path.join(output_dir, "res_line", "loading_percent.xlsx"),
    index_col=0
)
trafo_loading_df = pd.read_excel(
    os.path.join(output_dir, "res_trafo", "loading_percent.xlsx"),
    index_col=0
)


dm = pd.read_excel('./trial grid/res_bus/vm_pu.xlsx', index_col=0)  
dm.plot(legend=False)  # Disable legend  
plt.xlabel('time_step (Hours over a week)')  # Replace 'Your X Label' with your desired label  
plt.ylabel('res_bus voltage(p.u)')  # Replace 'Your Y Label' with your desired label  
plt.show()  


dm = pd.read_excel('./trial grid/res_trafo/loading_percent.xlsx', index_col=0)  
dm.plot(legend=False)  # Disable legend  
plt.xlabel("Hours over a week")  # Replace 'Your X Label' with your desired label  
plt.ylabel("Trafo loading %")  # Replace 'Your Y Label' with your desired label  
plt.show()  



# ****************************************************************************
# 8) Example Plots
#    (A) Bus Voltages over time (all buses, or just a subset if large)
#    (B) Line loading over time
#    (C) Transformer loading over time
# ****************************************************************************
title_size = 16
label_size = 14
tick_size  = 12
legend_size= 12

# (A) Plot bus voltages (all columns) in a single figure
plt.figure(figsize=(10,6), dpi=300)
for bus_col in bus_vm_df.columns:
    plt.plot(bus_vm_df[bus_col], label=f"Bus {bus_col}")
plt.title("All Bus Voltages Over Time", fontsize=title_size)
plt.xlabel("Hour over a week in winter", fontsize=label_size)
plt.ylabel("Voltage [p.u.]", fontsize=label_size)
plt.grid(None, linestyle='--', alpha=0.7)
plt.legend(None,fontsize=legend_size, ncol=3)  # adjust as needed
plt.tick_params(axis='both', which='major', labelsize=tick_size)
plt.tight_layout()
plt.savefig("AllBusVoltages.png", dpi=300)

plt.show()

# (B) Plot line loadings
plt.figure(figsize=(10,6), dpi=300)
for line_idx in line_loading_df.columns:
    plt.plot(line_loading_df[line_idx], label=f"Line {line_idx}")
plt.title("Line Loading (%) Over Time", fontsize=title_size)
plt.xlabel("Time Step", fontsize=label_size)
plt.ylabel("Loading [%]", fontsize=label_size)
plt.grid(None, linestyle='--', alpha=0.1)
plt.legend(None,fontsize=legend_size, ncol=3)

plt.tick_params(axis='both', which='major', labelsize=tick_size)
plt.tight_layout()

plt.savefig("LineLoading.png", dpi=300)
plt.show()

# (C) Plot transformer loading (if you only have one trafo, index=0)
trafo_idx = 0
if trafo_idx in trafo_loading_df.columns:
    plt.figure(figsize=(10,6), dpi=300)
    plt.plot(trafo_loading_df[trafo_idx], marker='s', label=f"Trafo {trafo_idx}")
    plt.title("Transformer Loading Over Time", fontsize=title_size)
    plt.xlabel("Time Step", fontsize=label_size)
    plt.ylabel("Loading [%]", fontsize=label_size)
    plt.legend(None,fontsize=legend_size)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=tick_size)
    plt.tight_layout()
    plt.savefig("TransformerLoading.png", dpi=300)
    
    plt.show()

# ****************************************************************************
# 9) Print the final results at the last time step

# ****************************************************************************
print("\nFinal Power Flow Results After Last Time Step:")
print("===============================================")
print("\nBus Results (net.res_bus):")
print(net.res_bus)
print("\nTransformer Results (net.res_trafo):")
print(net.res_trafo)
print("\nLoad Results (net.res_load):")
print(net.res_load)

print("\nAll time series results have been saved in the folder:", output_dir)
