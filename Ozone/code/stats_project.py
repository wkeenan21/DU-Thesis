import datetime
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#O3J = pd.read_csv(r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\csvs\today\ozoneJulyDenver.csv")
O3J = pd.read_csv(r"D:\Will_Git\DU-Thesis\Ozone\csvs\today\ozoneJulyDenver.csv")

O3J['date_local'] = pd.to_datetime(O3J['date_local'])
O3J = O3J.drop(axis=1, labels=['Unnamed: 0.1', 'Unnamed: 0','uncertaint', 'qualifier', 'LULC','clipDEM1km','LATITUDE_1', 'LONGITUD_1', 'ELEVATION_x','NAME_x','ELEVATION_y', 'NAME_y','HourlyDewPointTemperature','HourlyPrecipitation',
                                      'HourlyPresentWeatherType','HourlyPressureChange', 'HourlyPressureTendency','HourlySkyConditions','HourlySeaLevelPressure', 'HourlyStationPressure', 'HourlyVisibility', 'HourlyWindDirection','HourlyWindGustSpeed','WindEquipmentChangeDate'])

for index, row in O3J.iterrows():
    try:
        int(row['HourlyWindSpeed'])
    except:
        O3J.drop([index], inplace=True)
O3J['HourlyWindSpeed'] = pd.to_numeric(O3J['HourlyWindSpeed'])

for index, row in O3J.iterrows():
    if row['sample_mea'] < 0.005:
        O3J.drop([index], inplace=True)

# phaseShift = O3J.copy(deep=True)
# for i in range(4):
#     phaseShift.drop([i], inplace=True)
# phaseShift.reset_index(drop=True, inplace=True)
# phaseShift['shiftedO3'] = O3J['sample_mea'][0:-4]
# phaseShift = phaseShift[phaseShift['shiftedO3'].notna()]

# O3J['shifted_mea'] = np.zeros(shape=len(O3J))
# newTimes = pd.DataFrame(columns=['date_local','shifted_mea'])
# newTimes['date_local'] = np.zeros(len(O3J))
# newTimes['shifted_mea'] = np.zeros(len(O3J))

# O3J['date_local'] = pd.to_datetime(O3J['date_local'])
# newTimes['date_local'] = pd.to_datetime(newTimes['date_local'])
# newO3J = pd.merge(O3J, newTimes,  how='left', left_on='date_local', right_on='date_local')
#
# for index, row in O3J.iterrows():
#     newTime = row['date_local'] - datetime.timedelta(hours=4)
#     newTimes['date_local'][index] = newTime
#     newTimes['shifted_mea'][index] = row['sample_mea']
#     #O3J.loc[O3J['date_local'] == newTime, 'shifted_mea'] = row['sample_mea']
#
# just2 = phaseShift.copy(deep=True)
# for index, row in just2.iterrows():
#     if row['time_local'] != '14:00':
#         just2.drop([index], inplace=True)


#dfs for sites at just 2 pm
siteDfs = {}
for i in just2['site_will'].unique():
    siteDfs['site' + str(i)] = just2[just2['site_will'] == i]

#dfs for sites at all times
siteDfs_at = {}
for i in O3J['site_will'].unique():
    siteDfs_at['site' + str(i)] = O3J[O3J['site_will'] == i]

def runRegression(df, xvars, y):
    X = df[xvars]
    y = df[y]
    model = LinearRegression()
    model.fit(X, y)

    r_sq = model.score(X, y)
    print(f"coefficient of determination: {r_sq}")
    print('coefficients '+ str(model.coef_))
    print('intercept '+ str(model.intercept_))

def plotScatter(df, x, y, title='title'):
    plt.scatter(df[x], df[y])
    plt.title(title)
    plt.show()

plotScatter(just2, 'sample_mea', 'shiftedO3', title='Humidity vs Ozone')
runRegression(just2, xvars=['shiftedO3'], y='sample_mea')
for i in siteDfs:
    runRegression(siteDfs_at[i], ['HourlyWindSpeed', 'HourlyDryBulbTemperature', 'HourlyRelativeHumidity'], 'sample_mea')
    plotScatter(siteDfs_at[i], 'HourlyDryBulbTemperature', 'sample_mea', title=str(i))

runRegression(just2, xvars=['HourlyWindSpeed', 'HourlyWetBulbTemperature', 'HourlyRelativeHumidity'], y='sample_mea')

#just2.to_csv(r'C:\Users\willy\OneDrive - University of Denver\Stats\just2_ozone2.csv')
