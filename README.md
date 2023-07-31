# josephson_junction_analysis
This repository was created during an internship in the laboratory of the Institute of Micro and Nano Electronic Systems (IMS) at the Karlsruhe Institute of Technology (KIT).
In this lab course, we fabricated Josephson Junctions from the ground up, starting with depositing a Nb/Al-AlOx/Nb trilayer, over structuring the junctions and wiring, to depositing SiOx as an isolation layer. Afterwards, we measured a few of the produced junctions in the limited time we had left.
There were two types of Josephson Junctions measured, one with a meandering feed line and VIAs in the feed line and one type of josephson junctions with straight feedlines.
The VIAs were supposed to detach the junctions from stress induces from long feed lines. The comparison between the two types was then supposed to yield insights into the effect of stress induced by long feedlines.

The plotting into standalone HTML files is done with the plotting library Bokeh. Each HTML is included in the repository.

The following command imports the *.csv files from the measurements and produces plots for every current sweep. Additionally, some analyses were made to determine the critical temperature and the superconducting energy gap and more. To account for possible systematic errors in the fabrication process, the area of the junctions was corrected by considering the theoretical dependence of the normal resistance on area, as well as the critical current on area.

    python3 ./visualize_data_bokeh.py

The next script analyzed the difference between curved (short) and straight (long) feed lines. For this, area indepedent metrics were used to analyze differences between the two types. The standard deviations and mean values are calculated and a statistical test is performed.

    python3 ./visualize_curved_straight_bokeh.py

Lastly, measured voltage sweeps were analyzed and the subgap regions were compared to the subgap regions of the current sweeps.

    python3 ./visualize_vsweeps_bokeh.py