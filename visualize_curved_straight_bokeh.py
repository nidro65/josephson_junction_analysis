import pandas as pd
import os
from bokeh.plotting import figure, show, save, output_file
from bokeh.layouts import column, row
from bokeh.models import * # Paragraph, Div, ColumnDataSource, CDSView, GroupFilter
from bokeh.colors import RGB
from prefixed import Float

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



OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, "junction_isweep_data.csv")
with open(OUTPUT_FILEPATH, "r") as file:
    junctions_Isweep_df = pd.read_csv(file)


OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, "junction_isweep_filtered_out_data.csv")
with open(OUTPUT_FILEPATH, "r") as file:
    junctions_Isweep_filtered_out_df = pd.read_csv(file)

junctions_Isweep_df["type"] = junctions_Isweep_df["type"].str.replace("curved", "long")
junctions_Isweep_df["type"] = junctions_Isweep_df["type"].str.replace("straight", "short")
junctions_Isweep_filtered_out_df["type"] = junctions_Isweep_filtered_out_df["type"].str.replace("curved", "long")
junctions_Isweep_filtered_out_df["type"] = junctions_Isweep_filtered_out_df["type"].str.replace("straight", "short")

print("Junctions ISweep:")
print(junctions_Isweep_df.to_string())
print()
print("Junctions ISweep Filtered Out:")
print(junctions_Isweep_filtered_out_df.to_string())

junctions_Isweep_df["filtered"] = False
junctions_Isweep_filtered_out_df["filtered"] = True
total_df = pd.concat([junctions_Isweep_df[junctions_Isweep_filtered_out_df.columns], junctions_Isweep_filtered_out_df])
print(total_df.to_string())

MARKER_LIST = ["diamond", "hex", "inverted_triangle", "plus", "square", "star", "triangle"]


def plot_marker_for_chips(html_name: str, title: str, x: str, y: str, xlabel: str, ylabel: str, tooltips: list, legend_loc="top_left",
                          annotation_text=None, annotation_pos=None, show_in_browser=True):
    source = ColumnDataSource(total_df)
    # source_filtered_out = ColumnDataSource(junctions_Isweep_filtered_out_df)
    # chip_names = list(set(junctions_Isweep_df["chip_name"].unique()) | set(junctions_Isweep_filtered_out_df["chip_name"].unique()))
    chip_names = list(total_df["chip_name"].unique())

    OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, "comp", html_name)
    output_file(filename=OUTPUT_FILEPATH)

    p = figure(x_range=["short", "long"], title=title, width=1000, height=600, tooltips=tooltips)
    for m, chip_name in enumerate(chip_names):
        filter_chip = GroupFilter(column_name="chip_name", group=chip_name)

        chip_df = total_df[total_df["chip_name"] == chip_name]
        # chip_filtered_out_df = junctions_Isweep_filtered_out_df[junctions_Isweep_filtered_out_df["chip_name"]==chip_name]
        # plot points with different marker for every chip 
        if not chip_df[chip_df["filtered"] == False].empty:
            print("Test1")
            filter_bool = BooleanFilter(list(~total_df["filtered"]))
            print(filter_bool)
            view = CDSView(filter=(filter_chip & filter_bool))
            p.scatter(source=source, x=x, y=y, legend_label=f"{chip_name}", view=view, size=10, line_color=line_color, fill_color=None, marker=MARKER_LIST[m])

        # plot points that were filtered out muted
        if not chip_df[chip_df["filtered"] == True].empty:
            print("Test2")
            filter_bool = BooleanFilter(list(total_df["filtered"]))
            view = CDSView(filter=(filter_chip & filter_bool))
            p.scatter(source=source, x=x, y=y, legend_label=f"{chip_name}, filtered out", view=view, size=10, line_alpha=0.4, fill_alpha=0.4, line_color=line_color, fill_color=None, marker=MARKER_LIST[m])

    if annotation_text:
        if annotation_pos == "bottom_left":
            x_pos = 150
            y_pos = 40
            align = 'left'
        elif annotation_pos == "bottom_right":
            x_pos = 875
            y_pos = 40
            align = 'right'
        else:
            # still needs to be implemented
            x_pos = 0 + 1000/10
            y_pos = 40
            align = 'left'

        annotation = Label(x=x_pos, y=y_pos, x_units='screen', y_units='screen',
                        text=annotation_text, text_align=align)

        p.add_layout(annotation)

    # legend
    p.legend.location = legend_loc
    p.legend.click_policy="hide"
    # x-axis
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


TOOLTIPS = [
    # ("index", "$index"),
    # ("measurement", "@meas"),
    ("chip", "@chip_name"),
    ("T", "@T"),
    ("A", "@A"),
    # ("I_c", "@I_c_mean"),
    ("R_sg/R_N", "@R_sg_R_N"),
    ("I_C*R_N", "@I_c_R_N")
]

### --------- j_c ---------
p_j_c = plot_marker_for_chips(
    html_name="critical_current_densities.html",
    title="Comparison Critical Current Densities, Corrected Area",
    x="type",
    y="j_c_corr2_Ic",
    xlabel=r"\[ \text{Feedline Type}\]",
    ylabel=r"\[ \text{Critical Current Density } j_c \mathrm{~[A/cm^2]} \]",
    tooltips=TOOLTIPS,
    legend_loc="top_left",
    annotation_text=f"$$\Delta W = {Float(junctions_Isweep_df['deltaW_corr2_Ic'].iloc[0]*10**(-6)):.2h}m$$",
    annotation_pos="bottom_right",
    show_in_browser=True,
)

### --------- V_g ---------
p_V_g = plot_marker_for_chips(
    html_name="gap_voltage.html",
    title="Comparison Gap Voltage",
    x="type",
    y="V_g_mean",
    xlabel=r"\[ \text{Feedline Type}\]",
    ylabel=r"\[ \text{Gap Voltage } V_g \mathrm{~[V]}\]",
    tooltips=TOOLTIPS,
    legend_loc="top_left",
    show_in_browser=True,
)

### --------- R_sg / R_N ---------
p_Rsg_Rn = plot_marker_for_chips(
    html_name="R_sg_R_N.html",
    title="Comparison Subgap to Normal Resistance",
    x="type",
    y="R_sg_R_N",
    xlabel=r"\[ \text{Feedline Type}\]",
    ylabel=r"\[ \text{Quality Factor } R_{sg}/R_N \]",
    tooltips=TOOLTIPS,
    legend_loc="top_left",
    show_in_browser=True,
)

### --------- I_c * R_N ---------
p_Ic_Rn = plot_marker_for_chips(
    html_name="I_c_R_N.html",
    title="Comparison Ambegaokar-Baratoff Parameter",
    x="type",
    y="I_c_R_N",
    xlabel=r"\[ \text{Feedline Type}\]",
    ylabel=r"\[ \text{Ambegaokar-Baratoff Parameter } I_c \cdot R_N \mathrm{~[V]} \]",
    tooltips=TOOLTIPS,
    legend_loc="top_left",
    show_in_browser=True,
)