import pandas as pd
import os, sys
import matplotlib.pyplot as plt


PATH = os.path.join("NanoPr_w12_ChipAJ15+ArrayChipAK15", "CH1_JJ9x9_T4p22K_Isweep.csv")

df = pd.read_csv(PATH, sep=",", names=["V", "I", "t"], skiprows=1)

print(df.head()) 


fig = plt.figure()
plot1 = plt.plot(df["V"], df["I"])
plt.title("I-V CH1_JJ9x9_T4p22K_Isweep")
plt.xlabel("Voltage in V")
plt.ylabel("Current in A")
plt.show()
