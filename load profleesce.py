import pandas as pd
import matplotlib.pyplot as plt

# 1) Read your CSV file
csv_path = r"C:/Users/cusa/Downloads/data studiesload.csv"
df = pd.read_csv(csv_path)

# Optional: Confirm the columns
print("Columns found:", df.columns)

# 2) Compute sums for each column
sum_summer_total = df["total load summer(kW)"].sum()
sum_summer_avg   = df["Average summer(kW)"].sum()
sum_winter_total = df["total load winter(kW)"].sum()
sum_winter_avg   = df["Average(winter)(kW)"].sum()

# 3) Create the figure
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)  # high DPI for publication

# 4) Plot each column
hours = df["Hour"]
ax.plot(hours, df["total load summer(kW)"],  label="Total Summer",  color="red",   linewidth=1.5)
ax.plot(hours, df["Average summer(kW)"],    label="Average Summer", color="orange",linewidth=1.5)
ax.plot(hours, df["total load winter(kW)"], label="Total Winter",   color="blue",  linewidth=1.5)
ax.plot(hours, df["Average(winter)(kW)"],   label="Average Winter", color="green", linewidth=1.5)


# 4) Plot each column with markers
hours = df["Hour"]
ax.plot(hours, df["total load summer(kW)"],
        label="Total Summer",
        color="red",
        linewidth=1.5,
        marker='*',          # Circle marker
        markersize=8)

ax.plot(hours, df["Average summer(kW)"],
        label="Average Summer",
        color="grey",
        linewidth=1.5,
        marker='s',          # Square marker
        markersize=8)

ax.plot(hours, df["total load winter(kW)"],
        label="Total Winter",
        color="blue",
        linewidth=1.5,
        marker='^',          # Triangle marker
        markersize=8)

ax.plot(hours, df["Average(winter)(kW)"],
        label="Average Winter",
        color="black",
        linewidth=1.5,
        marker='D',          # Diamond marker
        markersize=8)


# 5) Increase axis label and tick label font sizes
ax.set_xlabel("Hour", fontsize=18)
ax.set_ylabel("kWh",   fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=18)

# Optional: Title and Legend (adjust font sizes as well if desired)
#ax.set_title("Summer & Winter Load Profiles (Total & Average)", fontsize=14)
ax.legend(loc="upper left", fontsize=13)
#ax.grid(True, linestyle="--", alpha=0.6)

# 6) Create a summary table with sums of each column
col_labels = ["Value"]
row_labels = [
    "(Summer Total)",
    "(Summer Average)",
    "(Winter Total)",
    "(Winter Average)"
]
table_vals = [
    [f"{sum_summer_total:,.2f}"],
    [f"{sum_summer_avg:,.2f}"],
    [f"{sum_winter_total:,.2f}"],
    [f"{sum_winter_avg:,.2f}"]
]

the_table = ax.table(
    cellText=table_vals,
    rowLabels=row_labels,
    colLabels=col_labels,
    loc="upper right", 
    cellLoc="right", 
    colWidths=[0.15]
)
# Make the table text a bit smaller
the_table.auto_set_font_size(False)
the_table.set_fontsize(18)
the_table.scale(1.0, 1.4)

# 7) Tidy layout
plt.tight_layout()

# 8) Show or save the figure
# plt.show()   # For interactive
plt.savefig("Summer_Winter_LoadProfile_withTable.png", bbox_inches="tight")
