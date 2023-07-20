import pandas as pd
import numpy as np
from numpy.polynomial import Polynomial
import os
import re
from prefixed import Float
from bokeh.plotting import figure, show, save, output_file
from bokeh.layouts import column, row
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
    if device_type == "straight":
        device_type = "long"
    elif device_type == "curved":
        device_type = "short"
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
    # title = f"Josephson Junction {junction_width}x{junction_length} {device_type} feedline, Wafer: w{wafer_num}, Chip: {chip_name}, T={temperature}"
    title_txt = f"Josephson Junction {junction_width}x{junction_length} {device_type} feedline, Wafer: w{wafer_num}, Chip: {chip_name}, "
    temp_txt = f"T={temperature}"
    title  = r"\[\text{" + title_txt + r"} " + temp_txt + r"\mathrm{K}\]"
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
