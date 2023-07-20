import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import os
from prefixed import Float
from bokeh.plotting import figure, show, save, output_file
# from bokeh.layouts import column, row
from bokeh.models import * # Paragraph, Div, ColumnDataSource, CDSView, GroupFilter
from bokeh.colors import RGB


MARKER_LIST = ["diamond", "hex", "inverted_triangle", "plus", "square", "star", "triangle"]
MEAN_MARKER_LIST = ["x", "asterisk", "cross", "y"]
DASH_STYLES = ["solid", "dashed", "dotted", "dotdash", "dashdot"]
CHIP_NAMES = ["AJ08", "AJ09", "AL08", "AI08"]

dirnames = [
    "01_NanoPr_w13_chipAJ08",
    "02_NanoPr_w13_chipAJ09",
    "03_NanoPr_w13_chipAL08", os.path.join("03_NanoPr_w13_chipAL08", "test_temp"),
    "04_NanoPr_w13_chipAI08", os.path.join("04_NanoPr_w13_chipAI08", "test_temp")]
#dirnames = ["NanoPr_w12_ChipAJ15+ArrayChipAK15"]  # old test data
OUTPUT_DIR = "output"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

THRESHOLD = 100 # Ohm (transition from subgap region to retrapping)
RANGE_I_C = 2e-5 # Volt (I_c is measured between 0 and RANGE_I_C)
RANGE_SUBGAP_MIN = 0.25e-3 # Volt (Line fit for R_sg between RANGE_SUBGAP_MIN and RANGE_SUBGAP_MAX)
RANGE_SUBGAP_MAX = 2.5e-3
RANGE_SUBGAP_NEG_MAX = -2.5e-3
VOLTAGE_R_SG = 2e-3
ARRAY_SIZE = 142

## Design
font_size_axis = "12pt"
font_size_major_ticks = "11pt"
font_style_axis = "normal"
font_size_title = "12pt"
font_style_title = "bold"
line_color = RGB(r=0, g=150, b=130)
line_width = 2.5

### TODO
# Arrays: (neglect)
# - other calculation of V_g
# - other calculation of I_trap for array 5x5
# - other calculation of subgap resistance 2mV not there
#
# Others:
# - draw points where we read params from
# - correct j_c for temperature
# - more data and box plot?
# - T_c bestimmen
# - klassifizieren warum welches device kaputt war

OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, "junction_data.csv")
with open(OUTPUT_FILEPATH, "r") as file:
    junctions_df = pd.read_csv(file)

# junctions_df = junctions_df[junctions_df["R_sg_R_N"]>10]  # filter out to low ratios of R_sg / R_N
# junctions_df = junctions_df[junctions_df["I_c_R_N"]>0.6e-3]  # filter out to low ratios ofI_c * R_N

# process and plot data with bokeh

## Filter out invalid data points
junctions_Isweep_df = junctions_df[junctions_df["meas"] == "Isweep"].copy()  # exclude IsweepTemp

# calc specific resistance rho = R_N * A
junctions_Isweep_df["R_spec"] = junctions_Isweep_df["R_N"] * junctions_Isweep_df["A"]
junctions_Isweep_df["1_over_A"] = 1/junctions_Isweep_df["A"]
junctions_Isweep_df["1_over_R_N"] = 1/junctions_Isweep_df["R_N"]


junctions_Isweep_filtered_out_df = junctions_Isweep_df[(junctions_Isweep_df["R_sg_R_N"]<=10) | (junctions_Isweep_df["I_c_R_N"]<=0.6e-3)]
print("filtered out junctions:")
print(junctions_Isweep_filtered_out_df.to_string())

junctions_Isweep_df = junctions_Isweep_df[junctions_Isweep_df["R_sg_R_N"]>10]  # filter out to low ratios of R_sg / R_N
junctions_Isweep_df = junctions_Isweep_df[junctions_Isweep_df["I_c_R_N"]>0.6e-3]  # filter out to low ratios ofI_c * R_N

keys = ["A", "j_c", "I_c_mean", "R_N", "R_spec", "R_sg", "R_sg_R_N", "I_c_R_N", "1_over_A", "1_over_R_N"]
junctions_mean_df = junctions_Isweep_df[keys].groupby("A").mean().reset_index(drop=False)

### ---- Correct area with best guess rho_0 ---- ###
best_rho_0_df = junctions_mean_df.loc[junctions_mean_df["A"].idxmax()]
# this r0 is most likely with correct area (largest area)
rho_0_corr1 = float(best_rho_0_df["R_spec"])
junctions_mean_df["A_corr1"] = (rho_0_corr1 / junctions_mean_df["R_N"])
junctions_Isweep_df["A_corr1"] = junctions_Isweep_df["A"]
for A in junctions_mean_df["A"].unique():
    junctions_Isweep_df.loc[junctions_Isweep_df["A"]==A, "A_corr1"] = float(junctions_mean_df.loc[junctions_mean_df["A"] == A, "A_corr1"].iloc[0])

junctions_mean_df["1_over_A_corr1"] = 1 / junctions_mean_df["A_corr1"]
# calc delta W for other points
# actual area should be smaller than wanted area --> deltaA should be negative
junctions_mean_df["deltaA_corr1"] = junctions_mean_df["A_corr1"]  - junctions_mean_df["A"]
junctions_mean_df["deltaW_corr1"] = junctions_mean_df["deltaA_corr1"].abs().pow(0.5)
junctions_mean_df = junctions_mean_df.assign(R_spec_corr1=rho_0_corr1)

junctions_Isweep_df["1_over_A_corr1"] = 1 / junctions_Isweep_df["A_corr1"]
junctions_Isweep_df["deltaA_corr1"] = junctions_Isweep_df["A_corr1"]  - junctions_Isweep_df["A"]
junctions_Isweep_df["deltaW_corr1"] = junctions_Isweep_df["deltaA_corr1"].abs().pow(0.5)
junctions_Isweep_df = junctions_Isweep_df.assign(R_spec_corr1=rho_0_corr1)

# correct critical currents and critical current density with the corrected area
junctions_mean_df["j_c_corr1"] = (junctions_mean_df["I_c_mean"] / junctions_mean_df["A_corr1"]) * 10**8
junctions_Isweep_df["j_c_corr1"] = (junctions_Isweep_df["I_c_mean"] / junctions_Isweep_df["A_corr1"]) * 10**8

### ---- next area correction try  1/R_N = 1/rho_0 * (A + deltaA)  with deltaA being constant (y-intercept point) ---- ###
# fit only valid for deltaW < W
def fit_1_over_R_N(A, deltaW, rho_0):
    W = np.sqrt(A)
    one_over_R_N = 1/rho_0 * np.power(W + deltaW, 2)
    return one_over_R_N

x_vals = junctions_Isweep_df["A"]
y_vals = junctions_Isweep_df["1_over_R_N"]
popt, pcov = curve_fit(fit_1_over_R_N, x_vals, y_vals)
deltaW_corr2_1 = popt[0]
rho_0_corr2_1 = popt[1]

def fit_I_c(A, deltaW, j_c):
    W = np.sqrt(A)
    I_c = j_c * np.power(W + deltaW, 2)
    return I_c

x_vals = junctions_Isweep_df["A"]
y_vals = junctions_Isweep_df["I_c_mean"]
popt, pcov = curve_fit(fit_I_c, x_vals, y_vals)
deltaW_corr2_2 = popt[0]
j_c_corr2_2 = popt[1]*10**8

# coeff_R_N_1 = np.polyfit(junctions_Isweep_df["A"], junctions_Isweep_df["1_over_R_N"], 1)
# rho_0_corr2 = 1/coeff_R_N_1[0]
# deltaA_corr2 = coeff_R_N_1[1] * rho_0_corr2
# deltaW_corr2_Ic = deltaA_corr2**0.5 if deltaA_corr2 >= 0 else -1*abs(deltaA_corr2)**0.5
# print(f"\n1/R_N fit coefficients: {coeff_R_N_1}")
# print(f"Inverse specific resistance: {coeff_R_N_1[0]} [1/(Ohm*um^2)]")
# print(f"Specific resistance: {rho_0_corr2} [Ohm*um^2]")
print("\nSecond Approach for Area Correction:")
print(f"Fit 1/R_N over A: dW={deltaW_corr2_1} [um], rho_0={rho_0_corr2_1} [Ohm*um^2]")
print(f"Fit I_c over A: dW={deltaW_corr2_2} [um], j_c={j_c_corr2_2} [A/cm^2]\n")

# junctions_Isweep_df = junctions_Isweep_df.assign(deltaA_corr2=deltaA_corr2)
junctions_Isweep_df = junctions_Isweep_df.assign(deltaW_corr2_Ic=deltaW_corr2_2)
junctions_Isweep_df["A_corr2_Ic"] = (junctions_Isweep_df["W"] + junctions_Isweep_df["deltaW_corr2_Ic"])**2  # formerly: junctions_Isweep_df["A"] + junctions_Isweep_df["deltaA_corr2"]
junctions_Isweep_df["1_over_A_corr2_Ic"] = 1 / junctions_Isweep_df["A_corr2_Ic"]
junctions_Isweep_df["j_c_corr2_Ic"] = junctions_Isweep_df["I_c_mean"] / junctions_Isweep_df["A_corr2_Ic"] * 10**8
junctions_Isweep_df["R_spec_corr2_Ic"] = junctions_Isweep_df["R_N"] / junctions_Isweep_df["A_corr2_Ic"]

junctions_Isweep_df = junctions_Isweep_df.assign(deltaW_corr2_Rn=deltaW_corr2_1)
junctions_Isweep_df["A_corr2_Rn"] = (junctions_Isweep_df["W"] + junctions_Isweep_df["deltaW_corr2_Rn"])**2  # formerly: junctions_Isweep_df["A"] + junctions_Isweep_df["deltaA_corr2"]
junctions_Isweep_df["1_over_A_corr2_Rn"] = 1 / junctions_Isweep_df["A_corr2_Rn"]
junctions_Isweep_df["j_c_corr2_Rn"] = junctions_Isweep_df["I_c_mean"] / junctions_Isweep_df["A_corr2_Rn"] * 10**8
junctions_Isweep_df["R_spec_corr2_Rn"] = junctions_Isweep_df["R_N"] / junctions_Isweep_df["A_corr2_Rn"]

keys = ["A", "A_corr2_Ic", "1_over_A_corr2_Ic", "j_c_corr2_Ic", "R_spec_corr2_Ic", "A_corr2_Rn", "1_over_A_corr2_Rn", "j_c_corr2_Rn", "R_spec_corr2_Rn"]
tmp_mean_df = junctions_Isweep_df[keys].groupby("A").mean().reset_index(drop=False)
junctions_mean_df[keys] = tmp_mean_df[keys]

junctions_Isweep_filtered_out_df = junctions_Isweep_filtered_out_df.assign(deltaW_corr2_Ic=deltaW_corr2_2)
junctions_Isweep_filtered_out_df["A_corr2_Ic"] = (junctions_Isweep_filtered_out_df["W"] + junctions_Isweep_filtered_out_df["deltaW_corr2_Ic"])**2
junctions_Isweep_filtered_out_df["1_over_A_corr2_Ic"] = 1 / junctions_Isweep_filtered_out_df["A_corr2_Ic"]
junctions_Isweep_filtered_out_df["j_c_corr2_Ic"] = junctions_Isweep_filtered_out_df["I_c_mean"] / junctions_Isweep_filtered_out_df["A_corr2_Ic"] * 10**8

junctions_Isweep_filtered_out_df = junctions_Isweep_filtered_out_df.assign(deltaW_corr2_Rn=deltaW_corr2_1)
junctions_Isweep_filtered_out_df["A_corr2_Rn"] = (junctions_Isweep_filtered_out_df["W"] + junctions_Isweep_filtered_out_df["deltaW_corr2_Rn"])**2
junctions_Isweep_filtered_out_df["1_over_A_corr2_Rn"] = 1 / junctions_Isweep_filtered_out_df["A_corr2_Rn"]
junctions_Isweep_filtered_out_df["j_c_corr2_Rn"] = junctions_Isweep_filtered_out_df["I_c_mean"] / junctions_Isweep_filtered_out_df["A_corr2_Rn"] * 10**8

print("junctions_mean_df:")
print(junctions_mean_df.to_string())
print()
print("junctions_Isweep_df:")
print(junctions_Isweep_df)

OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, "junction_isweep_data.csv")
junctions_Isweep_df.to_csv(OUTPUT_FILEPATH)

OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, "junction_isweep_filtered_out_data.csv")
junctions_Isweep_filtered_out_df.to_csv(OUTPUT_FILEPATH)

TOOLTIPS = [
    # ("index", "$index"),
    ("measurement", "@meas"),
    ("T", "@T"),
    ("I_c", "@I_c_mean"),
    ("R_sg/R_N", "@R_sg_R_N"),
    ("I_C*R_N", "@I_c_R_N")
]

def plot_marker_for_chips(html_name: str, title: str, xlist: list, ylist: list, fillcolorlist: list, xlabel: str, ylabel: str, tooltips: list, legend_loc="top_left",
                          show_in_browser=True, plot_mean=False, fit=None, bounds=None, plot_range="zero_max", fit_params=None, annotation_text=None, annotation_pos=None, use_temp_data=False):
    
    if use_temp_data:
        this_df = junctions_df
    else:
        this_df = junctions_Isweep_df
    
    source = ColumnDataSource(this_df)
    source_filtered_out = ColumnDataSource(junctions_Isweep_filtered_out_df)
    source_mean = ColumnDataSource(junctions_mean_df)

    min_x = 999999999
    max_x = 0

    OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, html_name)
    output_file(filename=OUTPUT_FILEPATH)

    p = figure(title=title, width=1000, height=600, tooltips=tooltips)
    filter_short=GroupFilter(column_name="type", group="short")
    filter_long=GroupFilter(column_name="type", group="long")
    for i, x in enumerate(xlist):
        if "corr" in x or "corr" in ylist[i]:
            # corrected_str = ", corrected area"
            corrected_str = ""
        else:
            corrected_str = ""

        for m, chip_name in enumerate(CHIP_NAMES):
            filter_chip = GroupFilter(column_name="chip_name", group=chip_name)
            chip_df = this_df[this_df["chip_name"] == chip_name]
            if not use_temp_data:
                chip_filtered_out_df = junctions_Isweep_filtered_out_df[junctions_Isweep_filtered_out_df["chip_name"]==chip_name]

            ### short feedlines
            # plot points with different marker for every chip 
            if not chip_df[chip_df["type"] == "short"].empty:
                view = CDSView(filter=(filter_short & filter_chip))
                if fillcolorlist[i] == "same":
                    fillcolor = "blue"
                else:
                    fillcolor = fillcolorlist[i]
                p.scatter(source=source, x=x, y=ylist[i], legend_label=f"{chip_name}, short feedline{corrected_str}", view=view, size=10, line_color="blue", fill_color=fillcolor, marker=MARKER_LIST[m])

            if not use_temp_data:
                # plot points that were filtered out muted
                # if not "corr" in x and not "corr" in ylist[i]:
                if not chip_filtered_out_df[chip_filtered_out_df["type"] == "short"].empty:
                    view = CDSView(filter=(filter_short & filter_chip))
                    if fillcolorlist[i] == "same":
                        fillcolor = "blue"
                    else:
                        fillcolor = fillcolorlist[i]
                    p.scatter(source=source_filtered_out, x=x, y=ylist[i], legend_label=f"{chip_name}, short feedline{corrected_str}, filtered out", view=view, size=10, line_alpha=0.3, fill_alpha=0.3, line_color="blue", fill_color=fillcolor, marker=MARKER_LIST[m])

            ### long feedlines
            # plot points with different marker for every chip 
            if not chip_df[chip_df["type"] == "long"].empty:
                view = CDSView(filter=(filter_long & filter_chip))
                if fillcolorlist[i] == "same":
                    fillcolor = "red"
                else:
                    fillcolor = fillcolorlist[i]
                p.scatter(source=source, x=x, y=ylist[i], legend_label=f"{chip_name}, long feedline{corrected_str}", view=view, size=10, line_color="red", fill_color=fillcolor, marker=MARKER_LIST[m])
            
            if not use_temp_data:
                # plot points that were filtered out muted
                # if not "corr" in x and not "corr" in ylist[i]:
                if not chip_filtered_out_df[chip_filtered_out_df["type"] == "long"].empty:
                    view = CDSView(filter=(filter_long & filter_chip))
                    if fillcolorlist[i] == "same":
                        fillcolor = "red"
                    else:
                        fillcolor = fillcolorlist[i]
                    p.scatter(source=source_filtered_out, x=x, y=ylist[i], legend_label=f"{chip_name}, long feedline{corrected_str}, filtered out", view=view, size=10, line_alpha=0.3, fill_alpha=0.3, line_color="red", fill_color=fillcolor, marker=MARKER_LIST[m])
    
            min_x = chip_df[x].min() if min_x > chip_df[x].min() else min_x
            max_x = chip_df[x].max() if max_x < chip_df[x].max() else max_x
            if not use_temp_data:
                if not "corr" in x and not "corr" in ylist[i]:
                    min_x = chip_filtered_out_df[x].min() if min_x > chip_filtered_out_df[x].min() else min_x
                    max_x = chip_filtered_out_df[x].max() if max_x < chip_filtered_out_df[x].max() else max_x

    
    if plot_mean:
        for i, x in enumerate(xlist):
            p.scatter(source=source_mean, x=x, y=ylist[i],legend_label = f"Mean {ylist[i]}", size=10, color="black", marker=MEAN_MARKER_LIST[i])
            # p.asterisk(source=source_mean, x="A_corr1", y="j_c_corr1",legend_label = "corrected means", size=10, color="black")


    # print(f"Min: {min_x}, max: {max_x}")
    if plot_range == "zero_max":
        x_range = np.arange(0,max_x*1.05,(max_x-0)/1000)
    elif plot_range == "min_max":
        x_range = np.arange(min_x*0.95,max_x*1.05,(max_x-min_x)/1000)
    else:
        # to be implemented
        x_range = np.arange(0,max_x*1.05,(max_x-0)/1000)

    if fit == "line":
        for i, x in enumerate(xlist):
            if "corr" in x or "corr" in ylist[i]:
                corrected_str = ", corrected area"
                # corrected_str = ""
            else:
                corrected_str = ""
            coeff = np.polyfit(this_df[x], this_df[ylist[i]], 1)
            poly1d_fn = np.poly1d(coeff)
            # x_range = np.arange(0,this_df[x].max()*1.05, this_df[x].max()/100)
            p.line(x=x_range, y=poly1d_fn(x_range), legend_label=f"Linear Fit {ylist[i]}{corrected_str}", color="black", line_dash=DASH_STYLES[i])
    elif fit == "line0intercept":
        for i, x in enumerate(xlist):
            if "corr" in x or "corr" in ylist[i]:
                corrected_str = ", corrected area"
            else:
                corrected_str = ""
            xvals = this_df[x].to_numpy()
            xvals = xvals[:,np.newaxis]
            yvals = np.array(this_df[ylist[i]])
            a, _, _, _ = np.linalg.lstsq(xvals, yvals, rcond=None)
            p.line(x=x_range, y=a*x_range, legend_label=f"Linear Fit Zero Intercept {ylist[i]}{corrected_str}", color="black", line_dash=DASH_STYLES[i])
    elif fit:
        for i, x in enumerate(xlist):
            if "corr" in x or "corr" in ylist[i]:
                corrected_str = ", corrected area"
            else:
                corrected_str = ""

            this_df_no_nans = this_df[this_df[x].notna()]
            this_df_no_nans = this_df_no_nans[this_df_no_nans[ylist[i]].notna()]
            xvals = this_df_no_nans[x]
            yvals = this_df_no_nans[ylist[i]]
            if bounds:
                popt, pcov = curve_fit(fit, xvals, yvals, bounds=bounds)
            else:
                popt, pcov = curve_fit(fit, xvals, yvals)
            fit_params.append(popt)
            p.line(x=x_range, y=fit(x_range, *popt), legend_label=f"Fit {ylist[i]}{corrected_str}", color="black", line_dash=DASH_STYLES[i])

    if annotation_text:
        if annotation_pos == "bottom_left":
            x = min_x + (max_x - min_x)/80
            y = 40
            align = 'left'
        elif annotation_pos == "bottom_right":
            x = max_x - (max_x - min_x)/80
            y = 40
            align = 'right'
        else:
            # still needs to be implemented
            x = min_x + (max_x - min_x)/80
            y = 40
            align = 'left'

        annotation = Label(x=x, y=y, x_units='data', y_units='screen',
                        text=annotation_text, text_align=align)

        p.add_layout(annotation)

    # legend
    p.legend.location = legend_loc
    p.legend.click_policy="hide"
    # x-axis
    # p.xaxis.axis_label = r"Junction Area \[A\] [µm²]"
    p.xaxis.axis_label = xlabel
    p.xaxis.axis_label_text_font_size = font_size_axis
    p.xaxis.axis_label_text_font_style = font_style_axis
    p.xaxis.major_label_text_font_size = font_size_major_ticks
    # y-axis
    p.yaxis.axis_label = ylabel
    p.yaxis.axis_label_text_font_size = font_size_axis
    p.yaxis.axis_label_text_font_style = font_style_axis
    p.yaxis.major_label_text_font_size = font_size_major_ticks
    # title
    p.title.text_font_size = font_size_title
    p.title.text_font_style = font_style_title

    if show_in_browser:
        show(p)
    else:
        save(p)

    return p


### --- J_C over Area ---
p_j_c = plot_marker_for_chips(
    html_name="critical_current_densities.html",
    title=r"\[\text{Critical Current Densities}\]",
    xlist=["A"],
    ylist=["j_c"],
    fillcolorlist=["same", None],
    xlabel=r"\[ \text{Junction Area } A \mathrm{~[\mu m^{2}]} \]",
    ylabel=r"\[ \text{Critical Current Density } j_c \mathrm{~[A/cm^2]} \]",
    tooltips=TOOLTIPS,
    plot_mean=True,
    legend_loc="top_right",
    show_in_browser=False,
)

# p_j_c_corr1 = plot_marker_for_chips(
#     html_name="critical_current_densities_corrected_area_1.html",
#     title="Critical Current Densities, corrected Area, Approach 1",
#     xlist=["A", "A_corr1"],
#     ylist=["j_c", "j_c_corr1"],
#     fillcolorlist=["same", None],
#     xlabel=r"\[ \text{Junction Area } A \mathrm{~[\mu m^{2}]} \]",
#     ylabel=r"\[ \text{Critical Current Density } j_c \mathrm{~[A/cm^2]} \]",
#     tooltips=TOOLTIPS,
#     plot_mean=True,
#     legend_loc="top_right",
#     show_in_browser=False,
# )

p_j_c_corr2 = plot_marker_for_chips(
    html_name="critical_current_densities_corrected_area_2.html",
    title=r"\[\text{Critical Current Densities, Area Correction with Critical Current}\]",
    xlist=["A", "A_corr2_Ic"],
    ylist=["j_c", "j_c_corr2_Ic"],
    fillcolorlist=["same", None],
    xlabel=r"\[ \text{Junction Area } A \mathrm{~[\mu m^{2}]} \]",
    ylabel=r"\[ \text{Critical Current Density } j_c \mathrm{~[A/cm^2]} \]",
    tooltips=TOOLTIPS,
    plot_mean=True,
    legend_loc="top_right",
    annotation_text="$$j_{c,corr} = " + f"{j_c_corr2_2:.2f}" + "A/cm^2,~\Delta W = " + f"{Float(deltaW_corr2_2*10**(-6)):.2h}m$$",  # f"$$j_c = {j_c:.2f}A/cm^2$$",
    annotation_pos="bottom_right",
    show_in_browser=True,
)

### --- I_C over Area ---
# x = junctions_Isweep_df["A"].to_numpy()
# x = x[:,np.newaxis]
# y = np.array(junctions_Isweep_df["I_c_mean"])
# a, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
# j_c_line0 = a[0]*10**8

# coeff = np.polyfit(junctions_Isweep_df["A"], junctions_Isweep_df["I_c_mean"], 1)
# j_c_line = coeff[0]*10**8
# deltaA_line = coeff[1]/coeff[0]
# deltaW_line = -1 * abs(deltaA_line)**0.5 if deltaA_line < 0 else deltaA_line**0.5
annot_str  =f"$$j_c = {j_c_corr2_2:.2f}A/cm^2,~\Delta W = {Float(deltaW_corr2_2*10**(-6)):.2h}m$$"

p_I_c = plot_marker_for_chips(
    html_name="critical_currents.html",
    title=r"\[\text{Critical Currents}\]",
    xlist=["A"],
    ylist=["I_c_mean"],
    fillcolorlist=["same"],
    xlabel=r"\[ \text{Junction Area } A \mathrm{~[\mu m^{2}]} \]",
    ylabel=r"\[ \text{Critical Current } I_c \mathrm{~[A]} \]",
    tooltips=TOOLTIPS,
    plot_mean=True,
    legend_loc="top_left",
    fit=fit_I_c, # "line0intercept",
    fit_params=[],
    annotation_text=annot_str,  # f"$$j_c = {j_c:.2f}A/cm^2$$",
    annotation_pos='bottom_right',
    show_in_browser=False,
)

### --- output I_c over Area, with corrected area --- ###
# x = junctions_Isweep_df["A_corr1"].to_numpy()
# x = x[:,np.newaxis]
# y = np.array(junctions_Isweep_df["I_c_mean"])
# a, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
# j_c_line0 = a[0]*10**8
# p_I_c_A_corr1 = plot_marker_for_chips(
#     html_name="critical_currents_corrected_area_1.html",
#     title="Critical Currents Corrected Area, corrected area, Approach 1",
#     xlist=["A", "A_corr1"],
#     ylist=["I_c_mean", "I_c_mean"],
#     fillcolorlist=["same", None],
#     xlabel=r"\[ \text{Junction Area } A \mathrm{~[\mu m^{2}]} \]",
#     ylabel=r"\[ \text{Critical Current } I_c \mathrm{~[A]} \]",
#     tooltips=TOOLTIPS,
#     plot_mean=True,
#     legend_loc="top_left",
#     fit="line0intercept",
#     # annotation_text="$$j_{c,corr} = " + f"{j_c_line0:.2f}" + "A/cm^2$$",  # f"$$j_c = {j_c:.2f}A/cm^2$$",
#     # annotation_pos="bottom_right",
#     show_in_browser=False,
# )

# x = junctions_Isweep_df["A"].to_numpy()
# x = x[:,np.newaxis]
# y = np.array(junctions_Isweep_df["I_c_mean"])
# a, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
# j_c_line0 = a[0]*10**8

fit_params = []
p_I_c_A_corr2 = plot_marker_for_chips(
    html_name="critical_currents_corrected_area_2.html",
    title=r"\[\text{Critical Currents, Area Correction with Critical Current}\]",
    xlist=["A", "A_corr2_Ic"],
    ylist=["I_c_mean", "I_c_mean"],
    fillcolorlist=["same", None],
    xlabel=r"\[ \text{Junction Area } A \mathrm{~[\mu m^{2}]} \]",
    ylabel=r"\[ \text{Critical Current } I_c \mathrm{~[A]} \]",
    tooltips=TOOLTIPS,
    plot_mean=True,
    legend_loc="top_left",
    fit=fit_I_c,
    fit_params=fit_params,
    # fit="line0intercept",
    annotation_text="$$j_{c,corr} = " + f"{j_c_corr2_2:.2f}" + "A/cm^2,~\Delta W = " + f"{Float(deltaW_corr2_2*10**(-6)):.2h}m$$",  # f"$$j_c = {j_c:.2f}A/cm^2$$",
    annotation_pos="bottom_right",
    show_in_browser=True,
)

# print(f"\nParams fit, deltaW={fit_params[0][0]}, j_C={fit_params[0][1]*10**(8)}")
# print(fit_params)

### --- output specific R over Area --- ###
# p_R_spec = plot_marker_for_chips(
#     html_name="specific_resistances.html",
#     title="Specific Resistances",
#     xlist=["A", "A_corr1"],
#     ylist=["R_spec", "R_spec_corr1"],
#     fillcolorlist=["same", None],
#     xlabel=r"\[ \text{Junction Area } A \mathrm{~[\mu m^{2}]} \]",
#     ylabel=r"\[\text{Specific Resistance } \rho_{0} \mathrm{~[\Omega ]}\]",
#     tooltips=TOOLTIPS,
#     plot_mean=True,
#     legend_loc="top_right",
#     show_in_browser=False,
# )

### --- output R_N over 1/Area --- ###
# p_R_N_invA_corr1 = plot_marker_for_chips(
#     html_name="normal_resistances_invA_corr1.html",
#     title="Normal Resistance over Inverted Area, corrected area, Approach 1",
#     xlist=["1_over_A", "1_over_A_corr1"],
#     ylist=["R_N", "R_N"],
#     fillcolorlist=["same", None],
#     xlabel=r"\[\text{Inverse Junction Area } 1/A \mathrm{~[1 / \mu m^{2}]}\]",
#     ylabel=r"\[\text{Normal Resistance } R_N \mathrm{~[\Omega ]}\]",
#     tooltips=TOOLTIPS,
#     plot_mean=True,
#     legend_loc="top_left",
#     fit="line",
#     show_in_browser=False,
# )

### --- output R_N over 1/Area --- ###
p_R_N_invA_corr2 = plot_marker_for_chips(
    html_name="normal_resistances_invA_corr2.html",
    title=r"\[\text{Normal Resistance over Inverted Area, Area Correction with Critical Current}\]",
    xlist=["1_over_A", "1_over_A_corr2_Ic"],
    ylist=["R_N", "R_N"],
    fillcolorlist=["same", None],
    xlabel=r"\[\text{Inverse Junction Area } 1/A \mathrm{~[1 / \mu m^{2}]}\]",
    ylabel=r"\[\text{Normal Resistance } R_N \mathrm{~[\Omega ]}\]",
    tooltips=TOOLTIPS,
    plot_mean=True,
    legend_loc="top_left",
    fit="line",
    show_in_browser=False,
)


### --- output 1/R_N over Area --- ###
# p_invR_N_A_corr1 = plot_marker_for_chips(
#     html_name="normal_resistances_invR_N_corr1.html",
#     title="Inverted Normal Resistance over Area, corrected area, Approach 1",
#     xlist=["A", "A_corr1"],
#     ylist=["1_over_R_N", "1_over_R_N"],
#     fillcolorlist=["same", None],
#     xlabel=r"\[ \text{Junction Area } A \mathrm{~[\mu m^{2}]} \]",
#     ylabel=r"\[\text{Inverse Normal Resistance } 1/R_N \mathrm{~[1/ \Omega ]}\]",
#     tooltips=TOOLTIPS,
#     plot_mean=True,
#     legend_loc="top_left",
#     fit="line",
#     show_in_browser=False,
# )

coeff = np.polyfit(junctions_Isweep_df["A"], junctions_Isweep_df["1_over_R_N"], 1)
poly1d_fn = np.poly1d(coeff)
x = 200
x_W = x**0.5
y = poly1d_fn(x)
rho_0 = 1/coeff[0]
dW = (rho_0 * y)**0.5 - x_W
dA = rho_0*y - x
print(f"\nNormal Line Fit for 1 over R_N:\nrho_0={1/coeff[0]}, intercept={coeff[1]}, deltaA={coeff[1]/coeff[0]}, deltaW={(abs(coeff[1]/coeff[0]))**0.5}")
print(f"x: {x}, x_W={x_W}, y={y}, dA={dA}, dW={dW}")
# poly1d_fn = np.poly1d(coeff)
# # x_range = np.arange(0,this_df[x].max()*1.05, this_df[x].max()/100)
# p.line(x=x_range, y=poly1d_fn(x_range), legend_label=f"Linear Fit {ylist[i]}{corrected_str}", color="black", line_dash=DASH_STYLES[i])

### --- output 1/R_N over Area --- ###
p_invR_N_A_corr2 = plot_marker_for_chips(
    html_name="normal_resistances_invR_N_corr2.html",
    title=r"\[\text{Inverted Normal Resistance over Area, Area Correction with Normal Resistance}\]",
    xlist=["A", "A_corr2_Rn"],
    ylist=["1_over_R_N", "1_over_R_N"],
    fillcolorlist=["same", None],
    xlabel=r"\[ \text{Junction Area } A \mathrm{~[\mu m^{2}]} \]",
    ylabel=r"\[\text{Inverse Normal Resistance } 1/R_N \mathrm{~[1/ \Omega ]}\]",
    tooltips=TOOLTIPS,
    plot_mean=True,
    legend_loc="top_left",
    fit=fit_1_over_R_N,
    fit_params=[],
    annotation_text=r"\[ \varrho_{0,corr} = " + f"{rho_0_corr2_1:.2f}" + r"\Omega \cdot \mu m^2,~\Delta W = " + f"{Float(deltaW_corr2_1*10**(-6)):.2h}" + r"m \]",  # f"$$j_c = {j_c:.2f}A/cm^2$$",
    annotation_pos="bottom_right",
    show_in_browser=True,
)

def one_over_x_fit(x, c1, c0):
    return c1/x + c0

### --- output R_N over Area --- ###
p_R_N_A_corr2 = plot_marker_for_chips(
    html_name="normal_resistances_R_N_corr2.html",
    title=r"\[\text{Normal Resistance over Area, Area Correction with Critical Current}\]",
    xlist=["A", "A_corr2_Ic"],
    ylist=["R_N", "R_N"],
    fillcolorlist=["same", None],
    xlabel=r"\[ \text{Junction Area } A \mathrm{~[\mu m^{2}]} \]",
    ylabel=r"\[\text{Normal Resistance } R_N \mathrm{~[\Omega ]}\]",
    tooltips=TOOLTIPS,
    plot_mean=True,
    legend_loc="top_right",
    fit=one_over_x_fit,
    plot_range="min_max",
    fit_params=[],
    annotation_text=r"\[\Delta W = " + f"{Float(deltaW_corr2_2*10**(-6)):.2h}m " + r"\]",  # f"$$j_c = {j_c:.2f}A/cm^2$$",
    annotation_pos="bottom_right",
    show_in_browser=True,
)

### --- Plot j_c over temperature --- ###
p_j_c_T = plot_marker_for_chips(
    html_name="critical_current_density_over_T.html",
    title=r"\[\text{Critical Current Density over Temperature}\]",
    xlist=["T"],
    ylist=["j_c"],
    use_temp_data=True,
    fillcolorlist=["same"],
    xlabel=r"\[\text{Temperature } T \mathrm{~[K]}\]",
    ylabel=r"\[ \text{Critical Current Density } j_c \mathrm{~[A/cm^2]} \]",
    tooltips=TOOLTIPS,
    plot_mean=False,
    legend_loc="top_right",
    show_in_browser=True,
)

### --- Plot V_g over temperature --- ###
def V_g_over_T(T, T_c, Delta_0_eV):
    # T: variable temperature
    # T_c, Delta_0: parameters for fit
    # approximation for temperature dependence of V_g
    Delta_T_eV = Delta_0_eV * (1 - (T/T_c)**4)**(2/3)
    V_g = Delta_T_eV * 2
    return V_g

junctions_df_no_nans = junctions_df[junctions_df["V_g_mean"].notna()]
x = junctions_df_no_nans["T"]
y = junctions_df_no_nans["V_g_mean"]
min_T_c = x.min()
max_T_c = 9.3 # K  (literature value: 9.26K, according to Wikipedia)
min_Delta_0_eV = y.min() / 2
max_Delta_0_eV = 3.05e-3 / 2  # literature gap voltage value 2.96e-3V at 4.2K, 3.05e-3V at 0K, should be well under this value
bounds = ([min_T_c, min_Delta_0_eV],[max_T_c, max_Delta_0_eV])
popt, pcov = curve_fit(V_g_over_T, x, y, bounds=bounds)

params = []

p_V_g_T = plot_marker_for_chips(
    html_name="gap_voltage_over_T.html",
    title=r"\[\text{Gap Voltage over Temperature}\]",
    xlist=["T"],
    ylist=["V_g_mean"],
    use_temp_data=True,
    fillcolorlist=["same"],
    xlabel=r"\[\text{Temperature } T \mathrm{~[K]}\]",
    ylabel=r"\[ \text{Gap Voltage } V_g \mathrm{~[V]} \]",
    tooltips=TOOLTIPS,
    # plot_mean=True,
    legend_loc="bottom_left",
    fit=V_g_over_T,
    bounds=bounds,
    plot_range="zero_max",
    fit_params=params,
    annotation_text=f"$$T_c = {popt[0]:.2f}K,~~\Delta_0 = {Float(popt[1]):.2h}eV$$",  # f"$$j_c = {j_c:.2f}A/cm^2$$",
    annotation_pos="bottom_right",
    show_in_browser=True,
)

# source_V_g_temp = ColumnDataSource(junctions_df)

# OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, "gap_voltage_over_T.html")
# output_file(filename=OUTPUT_FILEPATH)

# TOOLTIPS_V_G_T = [
#     # ("index", "$index"),
#     ("measurement", "@meas"),
#     ("T", "@T"),
#     ("length", "@L"),
#     ("j_c", "@j_c"),
#     ("R_sg/R_N", "@R_sg_R_N"),
#     ("I_C*R_N", "@I_c_R_N")
# ]


# # plotting
# p_V_g_T = figure(title=r"\[\text{Gap Voltage over Temperature}\]", width=1000, height=600) #, tooltips=TOOLTIPS_V_G_T)
# view_short = CDSView(filter=GroupFilter(column_name="type", group="short"))
# view_long = CDSView(filter=GroupFilter(column_name="type", group="long"))
# circ_short = p_V_g_T.circle(source=source_V_g_temp, x="T", y="V_g_mean",legend_label = "Short Feedline", view=view_short, size=10, color="blue")
# circ_long = p_V_g_T.circle(source=source_V_g_temp, x="T", y="V_g_mean",legend_label = "Long Feedline" ,view=view_long, size=10, color="red")
# circ_hover_tool = HoverTool(renderers=[circ_short, circ_long], tooltips=TOOLTIPS_V_G_T)
# p_V_g_T.add_tools(circ_hover_tool)

# # curve fits:
# junctions_df_no_nans = junctions_df[junctions_df["V_g_mean"].notna()]
# x = junctions_df_no_nans["T"]
# y = junctions_df_no_nans["V_g_mean"]
# min_T_c = x.min()
# max_T_c = 9.3 # K  (literature value: 9.26K, according to Wikipedia)
# min_Delta_0_eV = y.min() / 2
# max_Delta_0_eV = 3.05e-3 / 2  # literature gap voltage value 2.96e-3V at 4.2K, 3.05e-3V at 0K, should be well under this value
# bounds = ([min_T_c, min_Delta_0_eV],[max_T_c, max_Delta_0_eV])
# popt, pcov = curve_fit(V_g_over_T, x, y, bounds=bounds)
# x_range = np.arange(0, x.max()+1, 0.1)
# line_fit = p_V_g_T.line(x=x_range, y=V_g_over_T(x_range, *popt), legend_label="BCS fit", color="black")
# line_hover_tool = HoverTool(renderers=[line_fit], tooltips=[("T", "$x K"),("V_g", "$y V"),])
# p_V_g_T.add_tools(line_hover_tool)

# annotation1 = Label(x=40, y=40, x_units='screen', y_units='screen',
#                  text=f"$$T_c = {popt[0]:.2f}K,~~\Delta_0 = {Float(popt[1]):.2h}eV$$")

# # annotation2 = Label(x=40, y=60, x_units='screen', y_units='screen',
# #                  text="$$\dfrac{\Delta (T)}{\Delta (0)} \simeq \left[ 1-\left(\dfrac{T}{T_c}\right)^{4}\right]^{2/3}$$")

# p_V_g_T.add_layout(annotation1)
# # p_V_g_T.add_layout(annotation2)


# # legend
# p_V_g_T.legend.location = "top_right"
# p_V_g_T.legend.click_policy="hide"
# # x-axis
# p_V_g_T.xaxis.axis_label = r"\[\text{Temperature } T \mathrm{~[K]}\]"
# p_V_g_T.xaxis.axis_label_text_font_size = font_size_axis
# p_V_g_T.xaxis.axis_label_text_font_style = font_style_axis
# p_V_g_T.xaxis.major_label_text_font_size = font_size_major_ticks
# # y-axis
# p_V_g_T.yaxis.axis_label = r"\[ \text{Gap Voltage } V_g \mathrm{~[V]} \]"
# p_V_g_T.yaxis.axis_label_text_font_size = font_size_axis
# p_V_g_T.yaxis.axis_label_text_font_style = font_style_axis
# p_V_g_T.yaxis.major_label_text_font_size = font_size_major_ticks
# # title
# p_V_g_T.title.text_font_size = font_size_title
# p_V_g_T.title.text_font_style = font_style_title

# save(p_V_g_T)

