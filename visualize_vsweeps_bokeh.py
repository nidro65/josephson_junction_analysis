import pandas as pd
import numpy as np
from numpy.polynomial import Polynomial
import os, sys
import re
from prefixed import Float
from bokeh.plotting import figure, show, save, output_file
from bokeh.layouts import column, row
from bokeh.models import * # Paragraph, Div, ColumnDataSource, CDSView, GroupFilter
from bokeh.colors import RGB

dirnames = [
    "01_NanoPr_w13_chipAJ08",
    os.path.join("01_NanoPr_w13_chipAJ08", "other_meas"),
    "02_NanoPr_w13_chipAJ09",
    os.path.join("02_NanoPr_w13_chipAJ09", "other_meas"),
    "03_NanoPr_w13_chipAL08",
    # os.path.join("03_NanoPr_w13_chipAL08", "test_temp"),
    os.path.join("03_NanoPr_w13_chipAL08", "other_meas"),
    "04_NanoPr_w13_chipAI08",
    # os.path.join("04_NanoPr_w13_chipAI08", "test_temp"),
    os.path.join("04_NanoPr_w13_chipAI08", "other_meas"),]
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
    "name": [], "T": [], "meas": [], "type": [], "L": [], "W": [], "A": []}

data = {}
for filename, filepaths in merge_files_dict.items():
    print(f"\nGet Params for measurement {filepaths[0]}...")

    ### Get parameters
    # junction size
    match = re.match(r"^CH(\d+)\_JJ(\d+)x(\d+)\_?([^_]+)?\_T(\d+)p(\d+)K_(.+)$", filename)
    # print(match.groups())
    ch_nmb = int(match.groups()[0])
    junction_width = int(match.groups()[1])  # micrometer   TODO length or width?
    junction_length = int(match.groups()[2])  # micrometer
    junction_area_um = junction_width * junction_length
    junction_area_cm = junction_area_um * 10**(-8)
    device_type = match.groups()[3] # None for single junction, "array" for Array
    temperature = float(f"{match.groups()[4]}.{match.groups()[5]}")
    meas_type = match.groups()[6]
    print(f"Measurement: {meas_type} on Channel {ch_nmb} at {temperature}K")
    print(f"Junction Width: {junction_width}um, Length: {junction_length}um, Area: {junction_area_um}um²")
    print(f"Device Type: {device_type}")

    match = re.match(r"(\d+)_NanoPr_w(\d+)_chip([A-Z]{2}\d{2})", filepaths[0])
    # print(match.groups())
    meas_nmb = int(match.groups()[0])
    wafer = "w" + match.groups()[1]
    chip = match.groups()[2]
    print(f"Measurement #{meas_nmb}, wafer: {wafer}, chip: {chip}")

    if device_type == "array":
        continue

    df_main = None
    df_subgap = None
    # print(filepaths)
    for i, filepath in enumerate(filepaths):
        if i == 0:
            df_main = pd.read_csv(filepath, sep=",", names=["V", "I", "t"], skiprows=1)
        elif i == 1:
            df_subgap = pd.read_csv(filepath, sep=",", names=["V", "I", "t"], skiprows=1)
        else:
            print(f"!!! WARNING: more files than supported for {filename} !!!")
            break

    if df_subgap is not None:
        # merge subgap df with main df
        v_min_subgap = df_subgap["V"].min()
        v_max_subgap = df_subgap["V"].max()
        idx_v_max_subgap = df_subgap["V"].idxmax()
        df_subgap_1 = df_subgap[df_subgap.index <= idx_v_max_subgap]
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

    device_str = f"{wafer}_{chip}_{ch_nmb}"

    if not device_str in data:
        data[device_str] = {}

    if not meas_nmb in data[device_str]:
        data[device_str][meas_nmb] = {}

    if not meas_type in data[device_str][meas_nmb]:
        data[device_str][meas_nmb][meas_type] = [(df, filepaths[0])]
    else:
        data[device_str][meas_nmb][meas_type].append((df, filepaths[0]))

    if meas_type != "Vsweep":
        continue

    ### --- STORE DATA FOR EVERY JUNCTION --- ###
    junction_data["name"].append(filename)
    junction_data["T"].append(temperature)
    junction_data["meas"].append(meas_type)
    junction_data["type"].append(device_type)
    junction_data["L"].append(junction_length)
    junction_data["W"].append(junction_width)
    junction_data["A"].append(junction_area_um)

for device_str in data:
    for meas_nmb in data[device_str]:
        for meas_type in data[device_str][meas_nmb]:
            if meas_type != "Vsweep":
                continue

            for df, filepath in data[device_str][meas_nmb][meas_type]:
                print()

                ### BOKEH OUTPUT ###
                filename_ext = os.path.basename(filepath)
                filename = os.path.splitext(filename_ext)[0]
                OUTPUT_FILEPATH = os.path.join(os.path.dirname(filepath), filename+".html")
                output_file(filename=OUTPUT_FILEPATH)

                print(f"filename: {filename}, output_path: {OUTPUT_FILEPATH}")
                print(f"filepath: {filepath}")
                title = f"{filename}, Wafer: {device_str.split('_')[0]}, Chip: {device_str.split('_')[1]}"

                # create a new plot with a title and axis labels
                p = figure(title=title, width=1000, height=600)

                # add a line renderer with legend and line thickness
                p.line(df["V"], df["I"], legend_label="Vsweep", line_width=line_width, line_color=line_color)
                # plot only line of upwards sweep
                # p.line(df["V"][df["V"].index <= df["V"].idxmax()], df["I"][df["V"].index <= df["V"].idxmax()], legend_label="IVC", line_width=line_width, line_color=line_color)
                marker_plot = p.scatter(df["V"], df["I"], legend_label="Vsweep marker", marker="x", line_width=2, size=5, line_color="blue", alpha=0.8)
                marker_plot.visible = False

                for isweep_df, filepath in data[device_str][meas_nmb]["Isweep"]:
                    # add a line renderer with legend and line thickness
                    p.line(isweep_df["V"], isweep_df["I"], legend_label="Isweep", line_width=line_width, line_color=line_color, line_dash="dashed")
                    # plot only line of upwards sweep
                    # p.line(df["V"][df["V"].index <= df["V"].idxmax()], df["I"][df["V"].index <= df["V"].idxmax()], legend_label="IVC", line_width=line_width, line_color=line_color)
                    marker_plot = p.scatter(isweep_df["V"], isweep_df["I"], legend_label="Isweep marker", marker="x", line_width=2, size=5, line_color="red", alpha=0.8)
                    marker_plot.visible = False

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

                show(p, sizing_mode="stretch_both")

# store aggregated data of junctions in data frame
junctions_df = pd.DataFrame(data=junction_data)

# process and plot data with bokeh
print()
print(junctions_df)

OUTPUT_FILE= os.path.join(OUTPUT_DIR, "other_meas_data.csv")
junctions_df.to_csv(OUTPUT_FILE)
