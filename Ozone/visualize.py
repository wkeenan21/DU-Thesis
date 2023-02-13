import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

June15 = pd.read_csv(r'C:\Users\William.Keenan\Documents\GitHub\DU-Thesis\Ozone\Summer2021\ozone2021-06-15.csv')

sites = list(June15['site_number'].unique())

siteDfs = {}
for i in sites:
    siteDfs[str(i)] = June15[June15['site_number'] == i]

for i in sites:
    plt.plot(siteDfs[str(i)]['time_local'], siteDfs[str(i)]['sample_measurement'])


plt.show()