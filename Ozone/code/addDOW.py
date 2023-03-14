import pandas
import pandas as pd
import numpy as np
import os
import datetime as dt
#from PIL import Image
#from osgeo import gdal
from sklearn.linear_model import LinearRegression

O3J = pd.read_csv(r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\csvs\today\JulyPlus.csv")

# make new column for integer time
O3J['local_time_int'] = O3J['time_local']
O3J['local_time_int'] = O3J['local_time_int'].str.replace(':00', '')
O3J['local_time_int'] = pd.to_numeric(O3J['local_time_int'])

for i in range(len(O3J['date_local'])):
    O3J['date_local'][i] = O3J['date_local'][i].replace('0:00:00', O3J['time_local'][i])
O3J['date_local'] = pd.to_datetime(O3J['date_local'], format='%m/%d/%Y %H:%M')
O3J['day_of_week'] = O3J['date_local'].dt.day_name()

O3J['weekend'] = np.zeros(len(O3J))
O3J.loc[(O3J['day_of_week'] == 'Saturday') | (O3J['day_of_week'] == 'Sunday'), 'weekend'] = 1

# start changing the values
# this is if you care about hourly
# O3J.loc[(O3J['local_time_int'] < 6) & (O3J['weekend'] == 0), 'averageTra'] *= 0.3
# O3J.loc[(O3J['local_time_int'] >= 6) & (O3J['local_time_int'] < 10) & (O3J['weekend'] == 0), 'averageTra'] *= 2
# O3J.loc[(O3J['local_time_int'] >= 10) & (O3J['local_time_int'] < 15) & (O3J['weekend'] == 0), 'averageTra'] *= 1
# O3J.loc[(O3J['local_time_int'] >= 15) & (O3J['local_time_int'] < 19) & (O3J['weekend'] == 0), 'averageTra'] *= 2
# O3J.loc[(O3J['local_time_int'] >= 19) & (O3J['local_time_int'] < 23) & (O3J['weekend'] == 0), 'averageTra'] *= 0.5
# O3J.loc[(O3J['local_time_int'] == 23) & (O3J['weekend'] == 0), 'averageTra'] *= 0.4
#
# O3J.loc[(O3J['local_time_int'] < 7) | (O3J['local_time_int'] > 19) & (O3J['weekend'] == 1), 'averageTra'] *= 0.3
# O3J.loc[(O3J['local_time_int'] >= 7) | (O3J['local_time_int'] <= 19) & (O3J['weekend'] == 1), 'averageTra'] *= 1.5

# start changing the values
O3J.loc[O3J['local_time_int'] < 6, 'averageTra'] *= 0.1
O3J.loc[(O3J['local_time_int'] >= 6) & (O3J['local_time_int'] < 10), 'averageTra'] *= 2
O3J.loc[(O3J['local_time_int'] >= 10) & (O3J['local_time_int'] < 15), 'averageTra'] *= 1
O3J.loc[(O3J['local_time_int'] >= 15) & (O3J['local_time_int'] < 19), 'averageTra'] *= 2
O3J.loc[(O3J['local_time_int'] >= 19) & (O3J['local_time_int'] < 23), 'averageTra'] *= 0.5
O3J.loc[O3J['local_time_int'] == 23, 'averageTra'] *= 0.4

# this is only if you care about the weekend
# O3J.loc[O3J['weekend'] == 1, 'averageTra'] *= 0.8
# O3J.loc[O3J['weekend'] == 1, 'averageTra'] *= 1.2

#import weather data, filter for july
directory = r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\csvs\weather\Hourly2021"
wet = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    df = pd.read_csv(f)
    wet.append(df)

wet2 = []
for df in wet:
    # drop columns after wind speed
    columns = df.columns
    df.drop(labels=list(columns[24:-1]), axis=1, inplace=True)
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%dT%H:%M:%S')
    df_j = df[(df['DATE'] > '2021-07-01') & (df['DATE'] < '2021-08-01')]
    # drop unnecessary reports
    df_j = df_j[(df_j['REPORT_TYPE'] == 'FM-15')]
    # round minutes to closest hour
    df_j['DATE'] = df_j['DATE'].dt.round('H')
    df_j.drop_duplicates(subset=['DATE'], keep='first', inplace=True)
    wet2.append(df_j)

# site numbers are messed up sites 2 and 14 have multiple locations
siteCoords = []
for i in range(len(O3J['latitude'].unique())):
    siteCoords.append((O3J['latitude'].unique()[i], O3J['longitude'].unique()[i], i))

# add new column for my own site numbers
O3J['site_will'] = np.zeros(len(O3J))
for coord in siteCoords:
    O3J['site_will'] = np.where(O3J['latitude'] == coord[0], int(coord[2]), O3J['site_will'])

# downcast to integer
O3J['STATION'] = pd.to_numeric(O3J['STATION'], downcast='integer')

# concat weather dfs
weather = pd.DataFrame()
for df in wet2:
    if 'weather' in locals():
        weather = df.append(weather, ignore_index=True)
    else:
        weather = df

# merge weather and Ozone based on date and weather station columns
newO3J = pd.merge(O3J, weather,  how='left', left_on=['date_local','STATION'], right_on = ['DATE','STATION'])

# lots of missing data
for index, row in newO3J.iterrows():
    try:
        int(row['HourlyDryBulbTemperature'])
        int(row['HourlyWindSpeed'])
        int(row['HourlyRelativeHumidity'])
    except:
        newO3J.drop([index], inplace=True)

newO3J['HourlyDryBulbTemperature'] = pd.to_numeric(newO3J['HourlyDryBulbTemperature'])
newO3J['HourlyWindSpeed'] = pd.to_numeric(newO3J['HourlyWindSpeed'])
newO3J['HourlyRelativeHumidity'] = pd.to_numeric(newO3J['HourlyRelativeHumidity'])

just2 = newO3J.copy(deep=True)
for index, row in just2.iterrows():
    if row['time_local'] != '14:00' or row['sample_mea'] < 0.005:
        just2.drop([index], inplace=True)
just2.reset_index(drop=True, inplace=True)

phaseShift = just2.copy(deep=True)
phaseShift.drop([0], inplace=True)
phaseShift.reset_index(drop=True, inplace=True)
phaseShift['shifted_tra'] = just2['averageTra'][0:-2]
phaseShift = phaseShift[phaseShift['shifted_tra'].notna()]

oneStation = newO3J[newO3J['site_will'] == 0]

X = just2[['HourlyWindSpeed', 'HourlyDryBulbTemperature', 'HourlyRelativeHumidity']]
y = just2['sample_mea']

#plt.hist(just2['averageTra'])

model = LinearRegression()
model.fit(X, y)

r_sq = model.score(X, y)
print(f"coefficient of determination: {r_sq}")
print(model.coef_)
print(model.intercept_)