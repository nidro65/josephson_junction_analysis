import pandas as pd
import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit
import math
import os, sys
import re
from prefixed import Float
from bokeh.plotting import figure, show, save, output_file
from bokeh.layouts import column, row
from bokeh.models import * # Paragraph, Div, ColumnDataSource, CDSView, GroupFilter
from bokeh.colors import RGB


MARKER_LIST = ["diamond", "hex", "inverted_triangle", "plus", "square", "star", "triangle"]
MEAN_MARKER_LIST = ["x", "asterisk", "cross", "y"]
DASH_STYLES = ["solid", "dashed", "dotted", "dotdash", "dashdot"]

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


def line_from_slope_and_point(slope: float, x: float, y: float, x_list: list) -> list:
    m = float(slope)*10**(-8)
    c = y - m*x
    x_np = np.array(x_list)
    return m*x_np + c


merge_files_dict = {}
for dirname in dirnames:
    for x in os.listdir(dirname):
        FILEPATH = os.path.join(dirname, x)
        if os.path.isfile(FILEPATH):
            FILENAME = x
            FILENAME_NO_EXT, EXT = os.path.splitext(FILENAME)
            if EXT != ".csv":
                continue

            if not "_subgap" in FILENAME_NO_EXT:
                if FILENAME_NO_EXT in merge_files_dict:
                    merge_files_dict[FILENAME_NO_EXT].insert(0, FILEPATH)
                else:
                    merge_files_dict[FILENAME_NO_EXT] = [FILEPATH]
            else:
                FILENAME_NO_SUBGAP = FILENAME_NO_EXT.replace("_subgap", "")
                if FILENAME_NO_SUBGAP in merge_files_dict:
                    merge_files_dict[FILENAME_NO_SUBGAP].append(FILEPATH)
                else:
                    merge_files_dict[FILENAME_NO_SUBGAP] = [FILEPATH]

junction_data = {
    "name": [], "chip_name": [], "T": [], "meas": [], "type": [], "L": [], "W": [], "A": [], "V_g_pos": [], "V_g_neg": [], "V_g_mean": [],
    "I_c_pos": [], "I_c_neg": [], "I_c_mean": [], "j_c": [], "I_r_pos": [], "I_r_neg": [], "I_r_mean": [],
    "R_N": [], "R_sg": [], "R_sg_R_N": [], "I_c_R_N": []
    }

for filename, filepaths in merge_files_dict.items():
    print(f"\nGet Params for measurement {filepaths[0]}...")

    ### Get parameters
    # junction size
    match = re.match(r"^CH(\d+)\_JJ(\d+)x(\d+)\_?([^_]+)?\_T(\d+)p(\d+)K_(.+)$", filename)
    # print(match.groups())
    channel_number = int(match.groups()[0])
    junction_width = int(match.groups()[1])  # micrometer   TODO length or width?
    junction_length = int(match.groups()[2])  # micrometer
    junction_area_um = junction_width * junction_length
    junction_area_cm = junction_area_um * 10**(-8)
    device_type = match.groups()[3] # None for single junction, "array" for Array
    temperature = float(f"{match.groups()[4]}.{match.groups()[5]}")
    measurement_type = match.groups()[6]
    print(f"Measurement: {measurement_type} on Channel {channel_number} at {temperature}K")
    print(f"Junction Width: {junction_width}um, Length: {junction_length}um, Area: {junction_area_um}um²")
    print(f"Device Type: {device_type}")

    if device_type == "array":
        continue

    df_main = None
    df_subgap = None
    # print(filepaths)
    for i, filepath in enumerate(filepaths):
        if i == 0:
            df_main = pd.read_csv(filepath, sep=",", names=["V", "I", "t"], skiprows=1)
            match = re.search("\d+_NanoPr_w(\d+)_chip([\w]+)", filepath)
            if match:
                wafer_num = match.groups()[0]
                chip_name = match.groups()[1]
        elif i == 1:
            df_subgap = pd.read_csv(filepath, sep=",", names=["V", "I", "t"], skiprows=1)
        else:
            print(f"!!! WARNING: more files than supported for {filename} !!!")
            break
        # print(df.head())


    
    # p1 = figure(title=filename, sizing_mode="stretch_both")

    # # add a line renderer with legend and line thickness
    # p1.line(df_main["V"], df_main["I"], legend_label="IVC", line_width=line_width, line_color=line_color)
    # marker_plot_1 = p1.scatter(df_main["V"], df_main["I"], legend_label="Main Data", marker="x", line_width=2, line_color="blue")
    # marker_plot_2 = p1.scatter(df_subgap["V"], df_subgap["I"], legend_label="Subgap Data", marker="x", line_width=2, line_color="red")

    #show(p1)

    if df_subgap is not None:
        # merge subgap df with main df
        v_min_subgap = df_subgap["V"].min()
        v_max_subgap = df_subgap["V"].max()
        idx_v_max_subgap = df_subgap["V"].idxmax()
        df_subgap_1 = df_subgap[df_subgap.index <= idx_v_max_subgap]
        # print(df_subgap_1.to_string())
        # print(df_subgap_2.to_string())
        df_subgap_2 = df_subgap[df_subgap.index > idx_v_max_subgap]
        df_over_v_min_1 = df_main[df_main["V"] >= v_min_subgap]
        idx_over_v_min_1 = df_over_v_min_1.iloc[0].name
        idx_over_v_min_2 = df_main[df_main["V"] >= v_min_subgap].iloc[-1].name
        left_df_main_1 = df_main[df_main.index < idx_over_v_min_1]
        left_df_main_2 = df_main[df_main.index > idx_over_v_min_2]
        right_df_main = df_main[df_main["V"] > v_max_subgap]
        order = [left_df_main_1, df_subgap_1, right_df_main, df_subgap_2, left_df_main_2]
        df = pd.concat(order)
        df = df.reset_index()
        # print(df.to_string())
    else:
        df = df_main

    metrics = {}

    ### calc normal resistance
    R_N = (df["V"].max() - df["V"].min()) / (df["I"].max() - df["I"].min())
    print(f"Normal resistance R_N = {R_N:.3f} Ohm")
    metrics["R_N"] = ([df["V"].min(), df["V"].max()], [df["I"].min(), df["I"].max()])

    ### calc critical current
    # positive I_c
    df_I_c = df[df["V"].between(-RANGE_I_C, RANGE_I_C)]
    I_c_pos = df_I_c["I"].max()
    # negative I_c
    # print(df_I_c.to_string())
    I_c_neg = df_I_c["I"].min()
    # mean I_c
    I_c_mean = (I_c_pos - I_c_neg) / 2
    print(f"Critical Current I_c_pos = {Float(I_c_pos):!.3h}A I_c_neg = {Float(I_c_neg):!.3h}A I_c_mean = {Float(I_c_mean):!.3h}A")
    metrics["I_c+"] = (df_I_c.loc[df_I_c["I"].idxmax(), "V"], I_c_pos)
    metrics["I_c-"] = (df_I_c.loc[df_I_c["I"].idxmin(), "V"], I_c_neg)

    # calc j_c
    j_c = I_c_mean / junction_area_cm

    ### calc dynamic resistances
    # calc delta V's and delta I's
    df_withR = df.iloc[:-1].reset_index()
    df_withR2 = df.iloc[1:].reset_index()
    df_withR["V2"] = df_withR2["V"]
    df_withR["I2"] = df_withR2["I"]
    df_withR["deltaV"] = df_withR["V"] - df_withR["V2"]
    df_withR["deltaI"] = df_withR["I"] - df_withR["I2"]
    # calc differential R
    df_withR["R"] = (df_withR["V"] - df_withR["V2"]) / (df_withR["I"] - df_withR["I2"])
    
    ### --- calc retrapping current I_r ---
    ## calc positive I_r
    # choose range of I_r
    df_withR_backsweep = df_withR[df_withR.index >= df_withR["V"].idxmax()]
    df_withR_backsweep_over0V_under2mV = df_withR_backsweep[df_withR_backsweep["V"].between(-RANGE_I_C, RANGE_I_C, inclusive="both")]
    # find resistances under threshold, first one is I_r
    df_withR_withThreshold = df_withR_backsweep_over0V_under2mV[df_withR_backsweep_over0V_under2mV["R"] < THRESHOLD]
    I_r_pos = df_withR_withThreshold["I"].iloc[0]
    metrics["I_r+"] = (df_withR_withThreshold["V"].iloc[0], I_r_pos)

    ## calc negative I_r
    # choose range of I_r
    df_withR_upsweep = df_withR[df_withR.index <= df_withR["V"].idxmax()]
    df_withR_upsweep_under0V_overminus2mV = df_withR_upsweep[df_withR_upsweep["V"].between(-RANGE_I_C, RANGE_I_C, inclusive="both")]
    # print(df_withR_upsweep_under0V_overminus2mV.to_string())
    # find resistances under threshold, first one is I_r
    df_withR_withThreshold = df_withR_upsweep_under0V_overminus2mV[df_withR_upsweep_under0V_overminus2mV["R"] < THRESHOLD]
    I_r_neg = df_withR_withThreshold["I"].iloc[0]
    metrics["I_r-"] = (df_withR_withThreshold["V"].iloc[0], I_r_neg)

    ## calc mean of I_r
    I_r_mean = (I_r_pos - I_r_neg) / 2
    print(f"Retrapping current I_r_pos = {Float(I_r_pos):!.3h}A I_r_neg = {Float(I_r_neg):!.3h}A I_r_mean = {Float(I_r_neg):!.3h}A")

    ### --- Calc gap voltages V_g ---
    ## calc V_g_pos with 3rd order fit
    # only use upwards sweep
    df_V_g_pos = df_withR[df_withR["V"].index <= df_withR["V"].idxmax()]
    # look at data from RANGE_SUBGAP_MAX onwards (there are voltage jumps below, at I_c)
    df_V_g_pos = df_V_g_pos[df_V_g_pos["V"] >= RANGE_SUBGAP_MAX]
    # find voltage jump at end of transition to normal resistive state
    df_V_g_pos_cut = df_V_g_pos[df_V_g_pos.index <= df_V_g_pos["deltaV"].abs().idxmax()]
    # print(df_V_g_pos_cut.to_string())
    if df_V_g_pos_cut.shape[0] >= 4:
        # fit 3rd order polynomial to transition
        poly3d_fn_V_gap_pos = Polynomial.fit(df_V_g_pos_cut["V"], df_V_g_pos_cut["I"], 3)
        # derive two times to get inflection point
        poly2d_fn_V_gap_pos_deriv = poly3d_fn_V_gap_pos.deriv()
        poly1d_fn_V_gap_pos_deriv_deriv = poly2d_fn_V_gap_pos_deriv.deriv()
        # inflection point (V_g) is root of second derivation
        V_g_pos = poly1d_fn_V_gap_pos_deriv_deriv.roots()[0]
        # V_g = df_V_g["V"].iloc[0]  # old calculation
        metrics["V_g+"] = (V_g_pos, poly3d_fn_V_gap_pos(V_g_pos))
    else:
        print(f"!!! WARNING: V_g_pos could not be calculated for {filepaths[0]} !!!")
        V_g_pos = np.nan

    ## calc V_g_neg with 3rd order fit
    # only use downwards sweep
    df_V_g_neg = df_withR[df_withR["V"].index >= df_withR["V"].idxmax()]
    # look at data from RANGE_SUBGAP_MAX onwards (there are voltage jumps below, at I_c)
    df_V_g_neg = df_V_g_neg[df_V_g_neg["V"] <= RANGE_SUBGAP_NEG_MAX]
    # find voltage jump at end of transition to normal resistive state
    df_V_g_neg_cut = df_V_g_neg[df_V_g_neg.index <= df_V_g_neg["deltaV"].abs().idxmax()]
    # print(df_V_g_cut.to_string())

    if df_V_g_pos_cut.shape[0] >= 4:
        # fit 3rd order polynomial to transition
        poly3d_fn_V_gap_neg = Polynomial.fit(df_V_g_neg_cut["V"], df_V_g_neg_cut["I"], 3)
        # derive two times to get inflection point
        poly2d_fn_V_gap_neg_deriv = poly3d_fn_V_gap_neg.deriv()
        poly1d_fn_V_gap_neg_deriv_deriv = poly2d_fn_V_gap_neg_deriv.deriv()
        # inflection point (V_g) is root of second derivation
        V_g_neg = poly1d_fn_V_gap_neg_deriv_deriv.roots()[0]
        metrics["V_g-"] = (V_g_neg, poly3d_fn_V_gap_neg(V_g_neg))
    else:
        print(f"!!! WARNING: V_g_neg could not be calculated for {filepaths[0]} !!!")
        V_g_neg = np.nan

    ## calc v_g_mean
    V_g_mean = (V_g_pos - V_g_neg) / 2

    print(f"Gap voltage V_g_pos = {Float(V_g_pos):!.3h}V V_g_neg = {Float(V_g_neg):!.3h}V V_g_mean = {Float(V_g_mean):!.3h}V")



    ### --- Calc subgap resistance R_sg ---
    # line fit for 2mV value
    df_subgap_fit = df[df["V"].between(RANGE_SUBGAP_MIN, RANGE_SUBGAP_MAX, inclusive="both")]
    # print(df_subgap_fit.to_string())
    coeff = np.polyfit(df_subgap_fit["V"], df_subgap_fit["I"], 1)
    poly1d_fn = np.poly1d(coeff) 
    I_2mV = poly1d_fn(2e-3)
    R_sg = 2e-3 / I_2mV
    print(f"Subgap resistance R_sg = {R_sg:.3f} Ohm")
    metrics["R_sg_line"] = ([0, 2e-3, df["V"].max()], [0, I_2mV, (I_2mV/2e-3)*df["V"].max()])
    metrics["R_sg"] = (2e-3, I_2mV)

    ### --- calculate metrices ---
    R_sg_R_N = R_sg / R_N
    I_c_R_N = I_c_mean * R_N

    ### --- STORE DATA FOR EVERY JUNCTION --- ###
    junction_data["name"].append(filename)
    junction_data["chip_name"].append(chip_name)
    junction_data["T"].append(temperature)
    junction_data["meas"].append(measurement_type)
    junction_data["type"].append(device_type)
    junction_data["L"].append(junction_length)
    junction_data["W"].append(junction_width)
    junction_data["A"].append(junction_area_um)
    junction_data["V_g_pos"].append(V_g_pos)
    junction_data["V_g_neg"].append(V_g_neg)
    junction_data["V_g_mean"].append(V_g_mean)
    junction_data["I_c_pos"].append(I_c_pos)
    junction_data["I_c_neg"].append(I_c_neg)
    junction_data["I_c_mean"].append(I_c_mean)
    junction_data["I_r_pos"].append(I_r_pos)
    junction_data["I_r_neg"].append(I_r_neg)
    junction_data["I_r_mean"].append(I_r_mean)
    junction_data["j_c"].append(j_c)
    junction_data["R_N"].append(R_N)
    junction_data["R_sg"].append(R_sg)
    junction_data["R_sg_R_N"].append(R_sg_R_N)
    junction_data["I_c_R_N"].append(I_c_R_N)

    print()
    ### BOKEH OUTPUT ###
    OUTPUT_FILEPATH = os.path.join(os.path.dirname(filepath), filename+".html")
    output_file(filename=OUTPUT_FILEPATH)

    # create a new plot with a title and axis labels
    title = f"{filename}, Wafer: w{wafer_num}, Chip: {chip_name}"
    p = figure(title=title, width=1000, height=600)  # , sizing_mode="stretch_both")

    # add a line renderer with legend and line thickness
    p.line(df["V"], df["I"], legend_label="IVC", line_width=line_width, line_color=line_color)
    # plot only line of upwards sweep
    # p.line(df["V"][df["V"].index <= df["V"].idxmax()], df["I"][df["V"].index <= df["V"].idxmax()], legend_label="IVC", line_width=line_width, line_color=line_color)
    marker_plot = p.scatter(df["V"], df["I"], legend_label="IVC_marker", marker="x", line_width=2, line_color="blue")
    marker_plot.visible = False

    x = np.arange(RANGE_SUBGAP_MIN, RANGE_SUBGAP_MAX+0.25e-3, 0.25e-3)
    subgap_fit_plot = p.line(x, poly1d_fn(x), legend_label="Subgap Fit", line_color="black", line_width=2)
    subgap_fit_plot.visible = False

    x_3d = np.arange(df_V_g_pos_cut["V"].iloc[0], df_V_g_pos_cut["V"].iloc[-1], 1e-5)
    # print(x_3d)
    V_g_fit_plot = p.line(x_3d, poly3d_fn_V_gap_pos(x_3d), legend_label="V_Gap Fit", line_color="black", line_width=2)
    V_g_fit_plot.visible = False
    # p.line(df_withR["V"], df_withR["I"], legend_label="Current", line_width=2)
    # p.line(df_withR["V"], df_withR["R"], legend_label="Resistance", line_width=2, color="red")

    # plot points where metrics are readout
    count_lines = 0
    x_points = []
    y_points = []
    align = []
    names = []
    x_off = []
    y_off = []
    base = []
    for key, item in metrics.items():
        if type(item[1]) is list:
            # plot line
            p.line(x=item[0], y=item[1], line_color="black", legend_label="Metrics", line_dash=DASH_STYLES[count_lines], line_width=2, visible=True)
            count_lines += 1
        else:
            # plot point
            x_points.append(item[0])
            y_points.append(item[1])
            align.append("right") if item[1] > 0 else align.append("left")
            names.append(key)
            x_off.append(-5) if item[1] > 0 else x_off.append(+5)
            if key == "R_sg":
                y_off.append(12)
            else:
                y_off.append(5) if item[1] > 0 else y_off.append(-5)
            # base.append(key) if item[1] > 0 else align.append("right")


    label_source = ColumnDataSource(data={'x': x_points, 'y': y_points, 'name': names, 'align': align, 'x_off': x_off, 'y_off': y_off})
    p.scatter(source=label_source, x='x', y='y', marker="x", size=10, legend_label="Metrics", line_width=2, color="black", visible=True)
    labels = LabelSet(x='x', y='y', text='name', text_align='align', text_baseline='middle', x_offset='x_off', y_offset='y_off', source=label_source)
    p.add_layout(labels)

    # legend
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    # x-axis
    p.xaxis.axis_label = r"\[\text{Voltage } V \mathrm{~[V]}\]"
    p.xaxis.axis_label_text_font_size = font_size_axis
    p.xaxis.axis_label_text_font_style = font_style_axis
    p.xaxis.major_label_text_font_size = font_size_major_ticks
    # y-axis
    p.yaxis.axis_label = r"\[\text{Current } I \mathrm{~[A]}\]"
    p.yaxis.axis_label_text_font_size = font_size_axis
    p.yaxis.axis_label_text_font_style = font_style_axis
    p.yaxis.major_label_text_font_size = font_size_major_ticks
    # title
    p.title.text_font_size = font_size_title
    p.title.text_font_style = font_style_title

    # text output
    div_title = Div(text="<b>Calculated parameters:</b>", styles={'font-size': font_size_axis})
    div_size = Div(text=f"$$A = {junction_length}\cdot{junction_width} = {junction_area_um}$$" + r"$$\mathrm{\mu m^{2}}$$", styles={'font-size': font_size_axis})
    div_I_c_pos = Div(text=r"$$I_{c+}$$" + f"$$ = {Float(I_c_pos):!.3h}$$" + r"$$\mathrm{A}$$", styles={'font-size': font_size_axis})
    div_I_c_neg = Div(text=r"$$I_{c-}$$" + f"$$ = {Float(I_c_neg):!.3h}$$" + r"$$\mathrm{A}$$", styles={'font-size': font_size_axis})
    div_I_c_mean = Div(text=r"$$I_{c}$$" + f"$$ = {Float(I_c_mean):!.3h}$$" + r"$$\mathrm{A}$$", styles={'font-size': font_size_axis})
    div_I_r_pos = Div(text=r"$$I_{r+}$$" + f"$$ = {Float(I_r_pos):!.3h}$$" + r"$$\mathrm{A}$$", styles={'font-size': font_size_axis})
    div_I_r_neg = Div(text=r"$$I_{r-}$$" + f"$$ = {Float(I_r_neg):!.3h}$$" + r"$$\mathrm{A}$$", styles={'font-size': font_size_axis})
    div_I_r_mean = Div(text=r"$$I_{r}$$" + f"$$ = {Float(I_r_mean):!.3h}$$" + r"$$\mathrm{A}$$", styles={'font-size': font_size_axis})
    div_j_c = Div(text=f"$$j_c = {Float(j_c):!.3h}$$" + r"$$\mathrm{\dfrac{A}{cm^{2}}}$$", styles={'font-size': font_size_axis})
    div_V_g_pos = Div(text=r"$$V_{g+}$$" + f"$$ = {Float(V_g_pos):!.3h}$$" + r"$$\mathrm{V}$$", styles={'font-size': font_size_axis})
    div_V_g_neg = Div(text=r"$$V_{g-}$$" + f"$$ = {Float(V_g_neg):!.3h}$$" + r"$$\mathrm{V}$$", styles={'font-size': font_size_axis})
    div_V_g_mean = Div(text=r"$$V_{g}$$" + f"$$ = {Float(V_g_mean):!.3h}$$" + r"$$\mathrm{V}$$", styles={'font-size': font_size_axis})
    div_R_N = Div(text=f"$$R_N = {Float(R_N):!.3h}$$" + r"$$\mathrm{\Omega}$$", styles={'font-size': font_size_axis})
    div_R_sg = Div(text=r"$$R_{sg}$$" + f"$$ = {Float(R_sg):!.3h}$$" + r"$$\Omega$$", styles={'font-size': font_size_axis})
    div_R_sg_R_N = Div(text=r"$$\dfrac{R_{sg}}{R_N}$$" + f"$$ = {Float(R_sg_R_N):.3h}$$", styles={'font-size': font_size_axis})
    div_I_c_R_N = Div(text=r"$$I_{c} \cdot R_N$$" + f"$$ = {Float(I_c_R_N):.3h}$$" + r"$$\mathrm{V}$$", styles={'font-size': font_size_axis})

    # show the results
    row_I_r = row(div_I_r_pos, div_I_r_neg, div_I_r_mean)
    row_I_c = row(div_I_c_pos, div_I_c_neg, div_I_c_mean)
    row_V_g = row(div_V_g_pos, div_V_g_neg, div_V_g_mean)
    column_params = column( div_title, div_size, row_V_g, row_I_c, row_I_r, div_j_c, div_R_N, div_R_sg, div_R_sg_R_N, div_I_c_R_N )
    save(row(p, column_params, sizing_mode="stretch_both"))

# store aggregated data of junctions in data frame
junctions_df = pd.DataFrame(data=junction_data)

OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, "junction_data.csv")
junctions_df.to_csv(OUTPUT_FILEPATH)

# process and plot data with bokeh

## Filter out invalid data points
junctions_Isweep_df = junctions_df[junctions_df["meas"] == "Isweep"]  # exclude IsweepTemp

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
# deltaW_corr2 = deltaA_corr2**0.5 if deltaA_corr2 >= 0 else -1*abs(deltaA_corr2)**0.5
# print(f"\n1/R_N fit coefficients: {coeff_R_N_1}")
# print(f"Inverse specific resistance: {coeff_R_N_1[0]} [1/(Ohm*um^2)]")
# print(f"Specific resistance: {rho_0_corr2} [Ohm*um^2]")
print("\nSecond Approach for Area Correction:")
print(f"Fit 1/R_N over A: dW={deltaW_corr2_1} [um], rho_0={rho_0_corr2_1} [Ohm*um^2]")
print(f"Fit I_c over A: dW={deltaW_corr2_2} [um], j_c={j_c_corr2_2} [A/cm^2]\n")

# junctions_Isweep_df = junctions_Isweep_df.assign(deltaA_corr2=deltaA_corr2)
junctions_Isweep_df = junctions_Isweep_df.assign(deltaW_corr2=deltaW_corr2_1)
junctions_Isweep_df["A_corr2"] = (junctions_Isweep_df["W"] + junctions_Isweep_df["deltaW_corr2"])**2  # formerly: junctions_Isweep_df["A"] + junctions_Isweep_df["deltaA_corr2"]
junctions_Isweep_df["1_over_A_corr2"] = 1 / junctions_Isweep_df["A_corr2"]
junctions_Isweep_df["j_c_corr2"] = junctions_Isweep_df["I_c_mean"] / junctions_Isweep_df["A_corr2"] * 10**8
junctions_Isweep_df["R_spec_corr2"] = junctions_Isweep_df["R_N"] / junctions_Isweep_df["A_corr2"]

keys = ["A", "A_corr2", "1_over_A_corr2", "j_c_corr2", "R_spec_corr2"]
tmp_mean_df = junctions_Isweep_df[keys].groupby("A").mean().reset_index(drop=False)
junctions_mean_df[keys] = tmp_mean_df[keys]

junctions_Isweep_filtered_out_df = junctions_Isweep_filtered_out_df.assign(deltaW_corr2=deltaW_corr2_1)
junctions_Isweep_filtered_out_df["A_corr2"] = (junctions_Isweep_filtered_out_df["W"] + junctions_Isweep_filtered_out_df["deltaW_corr2"])**2
junctions_Isweep_filtered_out_df["j_c_corr2"] = junctions_Isweep_filtered_out_df["I_c_mean"] / junctions_Isweep_filtered_out_df["A_corr2"] * 10**8

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
                          show_in_browser=True, plot_mean=False, fit=None, bounds=None, fit_params=None, annotation_text=None, annotation_pos=None):
    source = ColumnDataSource(junctions_Isweep_df)
    source_filtered_out = ColumnDataSource(junctions_Isweep_filtered_out_df)
    source_mean = ColumnDataSource(junctions_mean_df)
    chip_names = list(set(junctions_Isweep_df["chip_name"].unique()) | set(junctions_Isweep_filtered_out_df["chip_name"].unique()))

    min_x = 0
    max_x = 0

    OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, html_name)
    output_file(filename=OUTPUT_FILEPATH)

    p = figure(title=title, width=1000, height=600, tooltips=tooltips)
    filter_curved=GroupFilter(column_name="type", group="curved")
    filter_straight=GroupFilter(column_name="type", group="straight")
    for i, x in enumerate(xlist):
        if "corr" in x or "corr" in ylist[i]:
            # corrected_str = ", corrected area"
            corrected_str = ""
        else:
            corrected_str = ""

        for m, chip_name in enumerate(chip_names):
            filter_chip = GroupFilter(column_name="chip_name", group=chip_name)
            chip_df = junctions_Isweep_df[junctions_Isweep_df["chip_name"] == chip_name]
            chip_filtered_out_df = junctions_Isweep_filtered_out_df[junctions_Isweep_filtered_out_df["chip_name"]==chip_name]

            ### long feedlines
            # plot points with different marker for every chip 
            if not chip_df[chip_df["type"] == "curved"].empty:
                view = CDSView(filter=(filter_curved & filter_chip))
                if fillcolorlist[i] == "same":
                    fillcolor = "blue"
                else:
                    fillcolor = fillcolorlist[i]
                p.scatter(source=source, x=x, y=ylist[i], legend_label=f"{chip_name}, long feedline{corrected_str}", view=view, size=10, line_color="blue", fill_color=fillcolor, marker=MARKER_LIST[m])

            # plot points that were filtered out muted
            if not "corr" in x and not "corr" in ylist[i]:
                if not chip_filtered_out_df[chip_filtered_out_df["type"] == "curved"].empty:
                    view = CDSView(filter=(filter_curved & filter_chip))
                    if fillcolorlist[i] == "same":
                        fillcolor = "blue"
                    else:
                        fillcolor = fillcolorlist[i]
                    p.scatter(source=source_filtered_out, x=x, y=ylist[i], legend_label=f"{chip_name}, long feedline{corrected_str}, filtered out", view=view, size=10, line_alpha=0.3, fill_alpha=0.3, line_color="blue", fill_color=fillcolor, marker=MARKER_LIST[m])

            ### short feedlines
            # plot points with different marker for every chip 
            if not chip_df[chip_df["type"] == "straight"].empty:
                view = CDSView(filter=(filter_straight & filter_chip))
                if fillcolorlist[i] == "same":
                    fillcolor = "red"
                else:
                    fillcolor = fillcolorlist[i]
                p.scatter(source=source, x=x, y=ylist[i], legend_label=f"{chip_name}, short feedline{corrected_str}", view=view, size=10, line_color="red", fill_color=fillcolor, marker=MARKER_LIST[m])
            
            # plot points that were filtered out muted
            if not "corr" in x and not "corr" in ylist[i]:
                if not chip_filtered_out_df[chip_filtered_out_df["type"] == "straight"].empty:
                    view = CDSView(filter=(filter_straight & filter_chip))
                    if fillcolorlist[i] == "same":
                        fillcolor = "red"
                    else:
                        fillcolor = fillcolorlist[i]
                    p.scatter(source=source_filtered_out, x=x, y=ylist[i], legend_label=f"{chip_name}, short feedline{corrected_str}, filtered out", view=view, size=10, line_alpha=0.3, fill_alpha=0.3, line_color="red", fill_color=fillcolor, marker=MARKER_LIST[m])
    
            min_x = chip_df[x].min() if min_x > chip_df[x].min() else min_x
            max_x = chip_df[x].max() if max_x < chip_df[x].max() else max_x
            if not "corr" in x and not "corr" in ylist[i]:
                min_x = chip_filtered_out_df[x].min() if min_x > chip_filtered_out_df[x].min() else min_x
                max_x = chip_filtered_out_df[x].max() if max_x < chip_filtered_out_df[x].max() else max_x

    
    # p.circle(source=source, x="A_corr1", y="j_c_corr1",legend_label = "Long Feedline (corrected area)", view=view_curved, size=10, line_color="blue", fill_color=None)
    # p.circle(source=source, x="A_corr1", y="j_c_corr1",legend_label = "Short Feedline (corrected area)" ,view=view_straight, size=10, line_color="red", fill_color=None)

    if plot_mean:
        for i, x in enumerate(xlist):
            p.scatter(source=source_mean, x=x, y=ylist[i],legend_label = f"Mean {ylist[i]}", size=10, color="black", marker=MEAN_MARKER_LIST[i])
            # p.asterisk(source=source_mean, x="A_corr1", y="j_c_corr1",legend_label = "corrected means", size=10, color="black")


    x_range = np.arange(0,max_x*1.05,(max_x-min_x)/1000)
    if fit == "line":
        for i, x in enumerate(xlist):
            if "corr" in x or "corr" in ylist[i]:
                corrected_str = ", corrected area"
                # corrected_str = ""
            else:
                corrected_str = ""
            coeff = np.polyfit(junctions_Isweep_df[x], junctions_Isweep_df[ylist[i]], 1)
            poly1d_fn = np.poly1d(coeff)
            # x_range = np.arange(0,junctions_Isweep_df[x].max()*1.05, junctions_Isweep_df[x].max()/100)
            p.line(x=x_range, y=poly1d_fn(x_range), legend_label=f"Linear Fit {ylist[i]}{corrected_str}", color="black", line_dash=DASH_STYLES[i])
    elif fit == "line0intercept":
        for i, x in enumerate(xlist):
            if "corr" in x or "corr" in ylist[i]:
                corrected_str = ", corrected area"
            else:
                corrected_str = ""
            xvals = junctions_Isweep_df[x].to_numpy()
            xvals = xvals[:,np.newaxis]
            yvals = np.array(junctions_Isweep_df[ylist[i]])
            a, _, _, _ = np.linalg.lstsq(xvals, yvals, rcond=None)
            p.line(x=x_range, y=a*x_range, legend_label=f"Linear Fit Zero Intercept {ylist[i]}{corrected_str}", color="black", line_dash=DASH_STYLES[i])
    elif fit:
        for i, x in enumerate(xlist):
            if "corr" in x or "corr" in ylist[i]:
                corrected_str = ", corrected area"
            else:
                corrected_str = ""
            xvals = junctions_Isweep_df[x]
            yvals = junctions_Isweep_df[ylist[i]]
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
    title="Critical Current Densities",
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

p_j_c_corr1 = plot_marker_for_chips(
    html_name="critical_current_densities_corrected_area_1.html",
    title="Critical Current Densities, corrected Area, Approach 1",
    xlist=["A", "A_corr1"],
    ylist=["j_c", "j_c_corr1"],
    fillcolorlist=["same", None],
    xlabel=r"\[ \text{Junction Area } A \mathrm{~[\mu m^{2}]} \]",
    ylabel=r"\[ \text{Critical Current Density } j_c \mathrm{~[A/cm^2]} \]",
    tooltips=TOOLTIPS,
    plot_mean=True,
    legend_loc="top_right",
    show_in_browser=False,
)

p_j_c_corr2 = plot_marker_for_chips(
    html_name="critical_current_densities_corrected_area_2.html",
    title="Critical Current Densities, corrected Area, Approach 2",
    xlist=["A", "A_corr2"],
    ylist=["j_c", "j_c_corr2"],
    fillcolorlist=["same", None],
    xlabel=r"\[ \text{Junction Area } A \mathrm{~[\mu m^{2}]} \]",
    ylabel=r"\[ \text{Critical Current Density } j_c \mathrm{~[A/cm^2]} \]",
    tooltips=TOOLTIPS,
    plot_mean=True,
    legend_loc="top_right",
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
    title="Critical Currents",
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
    show_in_browser=True,
)

### --- output I_c over Area, with corrected area --- ###
# x = junctions_Isweep_df["A_corr1"].to_numpy()
# x = x[:,np.newaxis]
# y = np.array(junctions_Isweep_df["I_c_mean"])
# a, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
# j_c_line0 = a[0]*10**8
p_I_c_A_corr1 = plot_marker_for_chips(
    html_name="critical_currents_corrected_area_1.html",
    title="Critical Currents Corrected Area, Approach 1",
    xlist=["A", "A_corr1"],
    ylist=["I_c_mean", "I_c_mean"],
    fillcolorlist=["same", None],
    xlabel=r"\[ \text{Junction Area } A \mathrm{~[\mu m^{2}]} \]",
    ylabel=r"\[ \text{Critical Current } I_c \mathrm{~[A]} \]",
    tooltips=TOOLTIPS,
    plot_mean=True,
    legend_loc="top_left",
    fit="line0intercept",
    # annotation_text="$$j_{c,corr} = " + f"{j_c_line0:.2f}" + "A/cm^2$$",  # f"$$j_c = {j_c:.2f}A/cm^2$$",
    # annotation_pos="bottom_right",
    show_in_browser=False,
)

# x = junctions_Isweep_df["A"].to_numpy()
# x = x[:,np.newaxis]
# y = np.array(junctions_Isweep_df["I_c_mean"])
# a, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
# j_c_line0 = a[0]*10**8

fit_params = []
p_I_c_A_corr2 = plot_marker_for_chips(
    html_name="critical_currents_corrected_area_2.html",
    title="Critical Currents Corrected Area, Approach 2",
    xlist=["A", "A_corr2"],
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

print(f"\nParams fit, deltaW={fit_params[0][0]}, j_C={fit_params[0][1]*10**(8)}")
print(fit_params)

### --- output specific R over Area --- ###
p_R_spec = plot_marker_for_chips(
    html_name="specific_resistances.html",
    title="Specific Resistances",
    xlist=["A", "A_corr1"],
    ylist=["R_spec", "R_spec_corr1"],
    fillcolorlist=["same", None],
    xlabel=r"\[ \text{Junction Area } A \mathrm{~[\mu m^{2}]} \]",
    ylabel=r"\[\text{Specific Resistance } \rho_{0} \mathrm{~[\Omega ]}\]",
    tooltips=TOOLTIPS,
    plot_mean=True,
    legend_loc="top_right",
    show_in_browser=False,
)

### --- output R_N over 1/Area --- ###
p_R_N_invA_corr1 = plot_marker_for_chips(
    html_name="normal_resistances_invA_corr1.html",
    title="Normal Resistance over Inverted Area, corrected area, Approach 1",
    xlist=["1_over_A", "1_over_A_corr1"],
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

### --- output R_N over 1/Area --- ###
p_R_N_invA_corr2 = plot_marker_for_chips(
    html_name="normal_resistances_invA_corr2.html",
    title="Normal Resistance over Inverted Area, corrected area, Approach 2",
    xlist=["1_over_A", "1_over_A_corr2"],
    ylist=["R_N", "R_N"],
    fillcolorlist=["same", None],
    xlabel=r"\[\text{Inverse Junction Area } 1/A \mathrm{~[1 / \mu m^{2}]}\]",
    ylabel=r"\[\text{Normal Resistance } R_N \mathrm{~[\Omega ]}\]",
    tooltips=TOOLTIPS,
    plot_mean=True,
    legend_loc="top_left",
    fit="line",
    show_in_browser=True,
)


### --- output 1/R_N over Area --- ###
p_invR_N_A_corr1 = plot_marker_for_chips(
    html_name="normal_resistances_invR_N_corr1.html",
    title="Inverted Normal Resistance over Area, corrected area, Approach 1",
    xlist=["A", "A_corr1"],
    ylist=["1_over_R_N", "1_over_R_N"],
    fillcolorlist=["same", None],
    xlabel=r"\[ \text{Junction Area } A \mathrm{~[\mu m^{2}]} \]",
    ylabel=r"\[\text{Inverse Normal Resistance } 1/R_N \mathrm{~[1/ \Omega ]}\]",
    tooltips=TOOLTIPS,
    plot_mean=True,
    legend_loc="top_left",
    fit="line",
    show_in_browser=False,
)
### --- output 1/R_N over Area --- ###
p_invR_N_A_corr2 = plot_marker_for_chips(
    html_name="normal_resistances_invR_N_corr2.html",
    title="Inverted Normal Resistance over Area, corrected area, Approach 2",
    xlist=["A", "A_corr2"],
    ylist=["1_over_R_N", "1_over_R_N"],
    fillcolorlist=["same", None],
    xlabel=r"\[ \text{Junction Area } A \mathrm{~[\mu m^{2}]} \]",
    ylabel=r"\[\text{Inverse Normal Resistance } 1/R_N \mathrm{~[1/ \Omega ]}\]",
    tooltips=TOOLTIPS,
    plot_mean=True,
    legend_loc="top_left",
    fit=fit_1_over_R_N,
    fit_params=[],
    annotation_text="\[ r_{0,corr} = " + f"{rho_0_corr2_1:.2f}" + "\Omega \cdot \mu m^2,~\Delta W = " + f"{Float(deltaW_corr2_1*10**(-6)):.2h}m \]",  # f"$$j_c = {j_c:.2f}A/cm^2$$",
    annotation_pos="bottom_right",
    show_in_browser=True,
)
# annotations:
# annotation1 = Label(x=40, y=40, x_units='screen', y_units='screen',
#                  text=f"$$T_c = {popt[0]:.2f}K,~~\Delta_0 = {Float(popt[1]):.2h}eV$$")




# source = ColumnDataSource(junctions_Isweep_df)
# source_mean = ColumnDataSource(junctions_mean_df)

# OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, "normal_resistances_invA.html")
# output_file(filename=OUTPUT_FILEPATH)

# TOOLTIPS_R_N_INV_A = [
#     # ("index", "$index"),
#     ("measurement", "@meas"),
#     ("T", "@T"),
#     ("j_c", "@j_c"),
#     ("R_sg/R_N", "@R_sg_R_N"),
#     ("I_C*R_N", "@I_c_R_N"),
#     ("A corrected", "@A_corr1"),
# ]
# p_R_N_invA = figure(title="Normal Resistances", sizing_mode="stretch_both", tooltips=TOOLTIPS_R_N_INV_A)
# view_curved = CDSView(filter=GroupFilter(column_name="type", group="curved"))
# view_straight = CDSView(filter=GroupFilter(column_name="type", group="straight"))
# p_R_N_invA.circle(source=source, x="1_over_A", y="R_N",legend_label = "Long Feedline", view=view_curved, size=10, color="blue")
# p_R_N_invA.circle(source=source, x="1_over_A", y="R_N",legend_label = "Short Feedline" ,view=view_straight, size=10, color="red")
# p_R_N_invA.scatter(source=source_mean, x="1_over_A", y="R_N",legend_label = "Mean Values", size=10, color="black", marker="x")
# p_R_N_invA.asterisk(source=source_mean, x="1_over_A_corr1", y="R_N",legend_label = "Corrected Mean Values", size=10, color="black")


# # TODO fit thru 0 intercept point, read out asymptot for bests guess rho_0

# x = junctions_Isweep_df["1_over_A"].to_numpy()
# x = x[:,np.newaxis]
# x_range = [0, x.max()]
# # print(x_range)
# y = np.array(junctions_Isweep_df["R_N"])
# rho_fitted, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
# p_R_N_invA.line(x=x_range, y=rho_fitted*x_range, legend_label = "1/x Fit", color="black", line_dash="dashed")
# print(f"Fitted specific resistance {rho_fitted}")

# # x = junctions_Isweep_df["A"].to_numpy()
# # x = x[:,np.newaxis]
# # y = np.array(junctions_Isweep_df["1_over_R_N"])
# # rho_fitted, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
# # p_R_N_invA.line(x=junctions_Isweep_df["A"], y=rho_fitted*junctions_Isweep_df["A"], legend_label = "Linear Fit", color="black", line_dash="dotted")
# # print(f"Fitted specific resistance {rho_fitted}")

# # coeff_R_N_over_invA = np.polyfit(junctions_mean_df["1_over_A"], junctions_mean_df["R_N"], 1)
# # poly1d_fn_R_N_over_invA = np.poly1d(coeff_R_N_over_invA)
# # p_R_N_invA.line(x=junctions_mean_df["1_over_A"], y=poly1d_fn_R_N_over_invA(junctions_mean_df["1_over_A"]), legend_label = "1/x Mean Fit", color="black")

# # coeff_R_N_over_invCorrA = np.polyfit(junctions_mean_df["1_over_A_corr1"], junctions_mean_df["R_N"], 1)
# # poly1d_fn_R_N_over_invCorrA = np.poly1d(coeff_R_N_over_invCorrA)
# # p_R_N_invA.line(x=junctions_mean_df["1_over_A_corr1"], y=poly1d_fn_R_N_over_invCorrA(junctions_mean_df["1_over_A_corr1"]), legend_label = "1/x Mean Fit area corrected", color="black", line_dash="dashed")

# # legend
# p_R_N_invA.legend.location = "top_left"
# p_R_N_invA.legend.click_policy="hide"
# # x-axis
# p_R_N_invA.xaxis.axis_label = r"\[\text{Inverse Junction Area } 1/A \mathrm{~[1 / \mu m^{2}]}\]"
# p_R_N_invA.xaxis.axis_label_text_font_size = font_size_axis
# p_R_N_invA.xaxis.axis_label_text_font_style = font_style_axis
# p_R_N_invA.xaxis.major_label_text_font_size = font_size_major_ticks
# p_R_N_invA.x_range.start = 0
# # y-axis
# p_R_N_invA.yaxis.axis_label = r"\[\text{Normal Resistance } R_N \mathrm{~[\Omega ]}\]"
# p_R_N_invA.yaxis.axis_label_text_font_size = font_size_axis
# p_R_N_invA.yaxis.axis_label_text_font_style = font_style_axis
# p_R_N_invA.yaxis.major_label_text_font_size = font_size_major_ticks
# # title
# p_R_N_invA.title.text_font_size = font_size_title
# p_R_N_invA.title.text_font_style = font_style_title

# save(p_R_N_invA)

# ### --- output R_N over 1/Area --- ###
# source = ColumnDataSource(junctions_Isweep_df)
# source_mean = ColumnDataSource(junctions_mean_df)

# OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, "normal_resistances_invR_N.html")
# output_file(filename=OUTPUT_FILEPATH)

# TOOLTIPS_INV_R_N_A = [
#     # ("index", "$index"),
#     ("measurement", "@meas"),
#     ("T", "@T"),
#     ("j_c", "@j_c"),
#     ("R_sg/R_N", "@R_sg_R_N"),
#     ("I_C*R_N", "@I_c_R_N"),
#     ("A corrected", "@A_corr1"),
# ]
# p_invR_N_A = figure(title="Normal Resistances", sizing_mode="stretch_both", tooltips=TOOLTIPS_INV_R_N_A)
# view_curved = CDSView(filter=GroupFilter(column_name="type", group="curved"))
# p_invR_N_A.circle(source=source, x="A", y="1_over_R_N",legend_label = "Long Feedline", view=view_curved, size=10, color="blue")
# view_straight = CDSView(filter=GroupFilter(column_name="type", group="straight"))
# p_invR_N_A.circle(source=source, x="A", y="1_over_R_N",legend_label = "Short Feedline" ,view=view_straight, size=10, color="red")
# p_invR_N_A.scatter(source=source_mean, x="A", y="1_over_R_N",legend_label = "Mean Values", size=10, color="black", marker="x")
# # p_invR_N_A.asterisk(source=source_mean, x="1_over_A_corr1", y="corrR_N",legend_label = "Corrected Mean Values", size=10, color="black")



# coeff_R_N_1 = np.polyfit(junctions_Isweep_df["A"], junctions_Isweep_df["1_over_R_N"], 1)
# poly1d_fn_R_N_1 = np.poly1d(coeff_R_N_1)
# As = np.arange(-1, junctions_Isweep_df["A"].max(), 1)
# p_invR_N_A.line(x=As, y=poly1d_fn_R_N_1(As), legend_label = "Linear Fit", color="black") #, line_dash="dotted")

# # coeff_R_N_1 = np.polyfit(junctions_Isweep_df["1_over_A"], junctions_Isweep_df["R_N"], 1)
# # poly1d_fn_R_N_1 = np.poly1d(coeff_R_N_1)
# # p_invR_N_A.line(x=junctions_Isweep_df["1_over_A"], y=poly1d_fn_R_N_1(junctions_Isweep_df["1_over_A"]), legend_label = "1/x Fit", color="black", line_dash="dotted")

# # coeff_R_N_over_invA = np.polyfit(junctions_mean_df["1_over_A"], junctions_mean_df["R_N"], 1)
# # poly1d_fn_R_N_over_invA = np.poly1d(coeff_R_N_over_invA)
# # p_invR_N_A.line(x=junctions_mean_df["1_over_A"], y=poly1d_fn_R_N_over_invA(junctions_mean_df["1_over_A"]), legend_label = "1/x Mean Fit", color="black")

# # coeff_R_N_over_invCorrA = np.polyfit(junctions_mean_df["1_over_A_corr1"], junctions_mean_df["R_N"], 1)
# # poly1d_fn_R_N_over_invCorrA = np.poly1d(coeff_R_N_over_invCorrA)
# # p_invR_N_A.line(x=junctions_mean_df["1_over_A_corr1"], y=poly1d_fn_R_N_over_invCorrA(junctions_mean_df["1_over_A_corr1"]), legend_label = "1/x Mean Fit area corrected", color="black", line_dash="dashed")


# annotation1 = Label(x=40, y=40, x_units='screen', y_units='screen',
#                  text=f"$$T_c = {popt[0]:.2f}K,~~\Delta_0 = {Float(popt[1]):.2h}eV$$")

# # annotation2 = Label(x=40, y=60, x_units='screen', y_units='screen',
# #                  text="$$\dfrac{\Delta (T)}{\Delta (0)} \simeq \left[ 1-\left(\dfrac{T}{T_c}\right)^{4}\right]^{2/3}$$")

# p_invR_N_A.add_layout(annotation1)

# # legend
# p_invR_N_A.legend.location = "top_left"
# p_invR_N_A.legend.click_policy="hide"
# # x-axis
# p_invR_N_A.xaxis.axis_label = r"\[\text{Junction Area } A \mathrm{~[\mu m^{2}]}\]"
# p_invR_N_A.xaxis.axis_label_text_font_size = font_size_axis
# p_invR_N_A.xaxis.axis_label_text_font_style = font_style_axis
# p_invR_N_A.xaxis.major_label_text_font_size = font_size_major_ticks
# p_invR_N_A.x_range.start = 0
# # y-axis
# p_invR_N_A.yaxis.axis_label = r"\[\text{Inverse Normal Resistance } 1/R_N \mathrm{~[1/ \Omega ]}\]"
# p_invR_N_A.yaxis.axis_label_text_font_size = font_size_axis
# p_invR_N_A.yaxis.axis_label_text_font_style = font_style_axis
# p_invR_N_A.yaxis.major_label_text_font_size = font_size_major_ticks
# # title
# p_invR_N_A.title.text_font_size = font_size_title
# p_invR_N_A.title.text_font_style = font_style_title

# show(p_invR_N_A)

### --- output R_N over Area --- ###
source = ColumnDataSource(junctions_Isweep_df)
source_mean = ColumnDataSource(junctions_mean_df)

OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, "normal_resistances.html")
output_file(filename=OUTPUT_FILEPATH)

TOOLTIPS_R_N = [
    # ("index", "$index"),
    ("measurement", "@meas"),
    ("T", "@T"),
    ("j_c", "@j_c"),
    ("R_sg/R_N", "@R_sg_R_N"),
    ("I_C*R_N", "@I_c_R_N"),
    ("A corrected", "@A_corr1"),
]
p_R_N = figure(title="Normal Resistances", width=1000, height=600, tooltips=TOOLTIPS_R_N)
view_curved = CDSView(filter=GroupFilter(column_name="type", group="curved"))
view_straight = CDSView(filter=GroupFilter(column_name="type", group="straight"))
p_R_N.circle(source=source, x="A", y="R_N",legend_label = "Long Feedline", view=view_curved, size=10, color="blue")
p_R_N.circle(source=source, x="A", y="R_N",legend_label = "Short Feedline" ,view=view_straight, size=10, color="red")
p_R_N.scatter(source=source_mean, x="A", y="R_N",legend_label = "Mean Values", size=10, color="black", marker="x")
p_R_N.asterisk(source=source_mean, x="A_corr1", y="R_N",legend_label = "Corrected Mean Values", size=10, color="black")


### 1/ x fit to uncorrected data
# A_1 = float(junctions_mean_df["A"].iloc[-2])
# A_2 = float(junctions_mean_df["A"].iloc[-1])
# R_1 = junctions_mean_df["R_N"].iloc[-2]
# R_2 = junctions_mean_df["R_N"].iloc[-1]
# # R_N = c1*A^(-1) + c0
# c1 = (R_1 - R_2) / ((1/A_1) - (1/A_2))
# c0 = R_1 - (c1/A_1)

# As = np.arange(float(junctions_mean_df["A"].iloc[0])-10, float(junctions_mean_df["A"].iloc[-1])+10, 0.1)
# Rs = c1/As + c0

# p_R_N.line(x=As, y=Rs, legend_label = "1/x Fit", color="black")

def one_over_x_fit(x, c1, c0):
    return c1/x + c0

# TODO: c1 ist rho_0??

x = junctions_Isweep_df["A"]
x_corr = junctions_Isweep_df["A_corr2"]
y = junctions_Isweep_df["R_N"]
# min_T_c = x.min()
# max_T_c = 9.3 # K  (literature value: 9.26K, according to Wikipedia)
# min_Delta_0_eV = y.min() / 2
# max_Delta_0_eV = 3.05e-3 / 2  # literature gap voltage value 2.96e-3V at 4.2K, 3.05e-3V at 0K, should be well under this value
# bounds = ([min_T_c, min_Delta_0_eV],[max_T_c, max_Delta_0_eV])
# print(bounds)
popt, pcov = curve_fit(one_over_x_fit, x, y) #, p0=[c1,c0]) #, bounds=bounds)
# print(f"\n{popt}")
# print(f"First guess values: {[c1,c0]}")
x_range = np.arange(float(junctions_Isweep_df["A"].min())-10, float(junctions_Isweep_df["A"].max())+10, 0.1)
line_fit_R_N = p_R_N.line(x=x_range, y=one_over_x_fit(x_range, *popt), legend_label="1/A fit", color="black")

popt_corr, pcov_corr = curve_fit(one_over_x_fit, x_corr, y)
line_fit_R_N_corr = p_R_N.line(x=x_range, y=one_over_x_fit(x_range, *popt_corr), legend_label="1/A fit corrected", color="black", line_dash="dashed")


line_R_N_hover_tool = HoverTool(renderers=[line_fit_R_N], tooltips=[("A", "$x um²"),("R_N", "$y Ohm"),])
p_R_N.add_tools(line_R_N_hover_tool)

# ## 1/ x fit to corrected data
# A_1 = float(junctions_mean_df["A_corr1"].iloc[-2])
# A_2 = float(junctions_mean_df["A_corr1"].iloc[-1])
# R_1 = junctions_mean_df["R_N"].iloc[-2]
# R_2 = junctions_mean_df["R_N"].iloc[-1]
# # R_N = c1*A^(-1) + c0
# c1 = (R_1 - R_2) / ((1/A_1) - (1/A_2))
# c0 = R_1 - (c1/A_1)

# # As = np.arange(float(junctions_mean_df["A_corr1"].iloc[0])-10, float(junctions_mean_df["A_corr1"].iloc[-1])+10, 0.1)
# Rs = c1/As + c0

# p_R_N.line(x=As, y=Rs,legend_label = "1/x Fit corrected", color="black", line_dash="dashed")

# legend
p_R_N.legend.location = "top_right"
p_R_N.legend.click_policy="hide"
# x-axis
p_R_N.xaxis.axis_label = r"\[\text{Junction Area } A \mathrm{~[\mu m^{2}]}\]"
p_R_N.xaxis.axis_label_text_font_size = font_size_axis
p_R_N.xaxis.axis_label_text_font_style = font_style_axis
p_R_N.xaxis.major_label_text_font_size = font_size_major_ticks
p_R_N.x_range.start = 0
# y-axis
p_R_N.yaxis.axis_label = r"\[\text{Normal Resistance } R_N \mathrm{~[\Omega ]}\]"
p_R_N.yaxis.axis_label_text_font_size = font_size_axis
p_R_N.yaxis.axis_label_text_font_style = font_style_axis
p_R_N.yaxis.major_label_text_font_size = font_size_major_ticks
# title
p_R_N.title.text_font_size = font_size_title
p_R_N.title.text_font_style = font_style_title

save(p_R_N)

### --- Plot j_c over temperature --- ###
source_temp = ColumnDataSource(junctions_df)

OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, "critical_current_density_over_T.html")
output_file(filename=OUTPUT_FILEPATH)

TOOLTIPS_J_C_T = [
    # ("index", "$index"),
    ("measurement", "@meas"),
    ("length", "@L"),
    ("j_c", "@j_c"),
    ("R_sg/R_N", "@R_sg_R_N"),
    ("I_C*R_N", "@I_c_R_N")
]
p_j_c_T = figure(title="Critical Current Density over Temperature", width=1000, height=600, tooltips=TOOLTIPS_J_C_T)
view_curved = CDSView(filter=GroupFilter(column_name="type", group="curved"))
p_j_c_T.circle(source=source_temp, x="T", y="j_c",legend_label = "Curved Feedline", view=view_curved, size=10, color="blue")
view_straight = CDSView(filter=GroupFilter(column_name="type", group="straight"))
p_j_c_T.circle(source=source_temp, x="T", y="j_c",legend_label = "Straight Feedline" ,view=view_straight, size=10, color="red")

# legend
p_j_c_T.legend.location = "top_left"
p_j_c_T.legend.click_policy="hide"
# x-axis
p_j_c_T.xaxis.axis_label = r"\[\text{Temperature } T \mathrm{~[K]}\]"
p_j_c_T.xaxis.axis_label_text_font_size = font_size_axis
p_j_c_T.xaxis.axis_label_text_font_style = font_style_axis
p_j_c_T.xaxis.major_label_text_font_size = font_size_major_ticks
# y-axis
p_j_c_T.yaxis.axis_label = r"\[ \text{Critical Current Density } j_c \mathrm{~[A/cm^2]} \]"
p_j_c_T.yaxis.axis_label_text_font_size = font_size_axis
p_j_c_T.yaxis.axis_label_text_font_style = font_style_axis
p_j_c_T.yaxis.major_label_text_font_size = font_size_major_ticks
# title
p_j_c_T.title.text_font_size = font_size_title
p_j_c_T.title.text_font_style = font_style_title

save(p_j_c_T)

### --- Plot V_g over temperature --- ###

def V_g_over_T(T, T_c, Delta_0_eV):
    # T: variable temperature
    # T_c, Delta_0: parameters for fit
    # approximation for temperature dependence of V_g
    Delta_T_eV = Delta_0_eV * (1 - (T/T_c)**4)**(2/3)
    V_g = Delta_T_eV * 2
    return V_g

source_V_g_temp = ColumnDataSource(junctions_df)

OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, "gap_voltage_over_T.html")
output_file(filename=OUTPUT_FILEPATH)

TOOLTIPS_V_G_T = [
    # ("index", "$index"),
    ("measurement", "@meas"),
    ("T", "@T"),
    ("length", "@L"),
    ("j_c", "@j_c"),
    ("R_sg/R_N", "@R_sg_R_N"),
    ("I_C*R_N", "@I_c_R_N")
]


# plotting
p_V_g_T = figure(title="Gap Voltage over Temperature", width=1000, height=600) #, tooltips=TOOLTIPS_V_G_T)
view_curved = CDSView(filter=GroupFilter(column_name="type", group="curved"))
view_straight = CDSView(filter=GroupFilter(column_name="type", group="straight"))
circ_curved = p_V_g_T.circle(source=source_V_g_temp, x="T", y="V_g_mean",legend_label = "Curved Feedline", view=view_curved, size=10, color="blue")
circ_straight = p_V_g_T.circle(source=source_V_g_temp, x="T", y="V_g_mean",legend_label = "Straight Feedline" ,view=view_straight, size=10, color="red")
circ_hover_tool = HoverTool(renderers=[circ_curved, circ_straight], tooltips=TOOLTIPS_V_G_T)
p_V_g_T.add_tools(circ_hover_tool)

# curve fits:
junctions_df_no_nans = junctions_df[junctions_df["V_g_mean"].notna()]
x = junctions_df_no_nans["T"]
y = junctions_df_no_nans["V_g_mean"]
min_T_c = x.min()
max_T_c = 9.3 # K  (literature value: 9.26K, according to Wikipedia)
min_Delta_0_eV = y.min() / 2
max_Delta_0_eV = 3.05e-3 / 2  # literature gap voltage value 2.96e-3V at 4.2K, 3.05e-3V at 0K, should be well under this value
bounds = ([min_T_c, min_Delta_0_eV],[max_T_c, max_Delta_0_eV])
popt, pcov = curve_fit(V_g_over_T, x, y, bounds=bounds)
x_range = np.arange(0, x.max()+1, 0.1)
line_fit = p_V_g_T.line(x=x_range, y=V_g_over_T(x_range, *popt), legend_label="BCS fit", color="black")
line_hover_tool = HoverTool(renderers=[line_fit], tooltips=[("T", "$x K"),("V_g", "$y V"),])
p_V_g_T.add_tools(line_hover_tool)

annotation1 = Label(x=40, y=40, x_units='screen', y_units='screen',
                 text=f"$$T_c = {popt[0]:.2f}K,~~\Delta_0 = {Float(popt[1]):.2h}eV$$")

# annotation2 = Label(x=40, y=60, x_units='screen', y_units='screen',
#                  text="$$\dfrac{\Delta (T)}{\Delta (0)} \simeq \left[ 1-\left(\dfrac{T}{T_c}\right)^{4}\right]^{2/3}$$")

p_V_g_T.add_layout(annotation1)
# p_V_g_T.add_layout(annotation2)


# legend
p_V_g_T.legend.location = "top_right"
p_V_g_T.legend.click_policy="hide"
# x-axis
p_V_g_T.xaxis.axis_label = r"\[\text{Temperature } T \mathrm{~[K]}\]"
p_V_g_T.xaxis.axis_label_text_font_size = font_size_axis
p_V_g_T.xaxis.axis_label_text_font_style = font_style_axis
p_V_g_T.xaxis.major_label_text_font_size = font_size_major_ticks
# y-axis
p_V_g_T.yaxis.axis_label = r"\[ \text{Gap Voltage } V_g \mathrm{~[V]} \]"
p_V_g_T.yaxis.axis_label_text_font_size = font_size_axis
p_V_g_T.yaxis.axis_label_text_font_style = font_style_axis
p_V_g_T.yaxis.major_label_text_font_size = font_size_major_ticks
# title
p_V_g_T.title.text_font_size = font_size_title
p_V_g_T.title.text_font_style = font_style_title

save(p_V_g_T)

