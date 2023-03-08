import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

O3J = pd.read_csv(r"D:\Will_Git\DU-Thesis\Ozone\csvs\today\ozoneJulyDenver.csv")

for index, row in O3J.iterrows():
    try:
        int(row['HourlyWindSpeed'])
    except:
        O3J.drop([index], inplace=True)
O3J['HourlyWindSpeed'] = pd.to_numeric(O3J['HourlyWindSpeed'])

just2 = O3J.copy(deep=True)

for index, row in just2.iterrows():
    if row['time_local'] != '14:00' or row['sample_mea'] < 0.005:
        just2.drop([index], inplace=True)

import matplotlib.pyplot as plt
plt.scatter(just2['HourlyWindSpeed'],just2['sample_mea'])
plt.show()

X = just2[['HourlyWindSpeed','HourlyDryBulbTemperature']]
y = just2['sample_mea']

plt.hist(just2['averageTra'])

model = LinearRegression()
model.fit(X, y)

r_sq = model.score(X, y)
print(f"coefficient of determination: {r_sq}")
print(model.coef_)
print(model.intercept_)