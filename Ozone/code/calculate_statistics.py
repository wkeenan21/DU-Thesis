import pandas as pd
import numpy as np
import os
#from PIL import Image
from osgeo import gdal
from sklearn.linear_model import LinearRegression
# import data
# OzoneJuly = pd.read_csv (r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\csvs\today\JulyOzone.csv")
# ds = gdal.Open(r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\gisData\LULC\rc_rc_clip.tif", gdal.GA_ReadOnly)
# LULC = np.array(ds.GetRasterBand(1).ReadAsArray())
# geoTransform = ds.GetGeoTransform()
# minx = geoTransform[0]
# maxy = geoTransform[3]
# maxx = minx + geoTransform[1] * ds.RasterXSize
# miny = maxy + geoTransform[5] * ds.RasterYSize
# extent = [miny, minx, maxy, maxx]
#
# latitudes = OzoneJuly['latitude'].unique()
# longitudes = OzoneJuly['longitude'].unique()
# sites = OzoneJuly['site_number'].unique()


O3J = pd.read_csv(r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\csvs\today\JulyPlusWeather.csv")

# make new column for integer time
O3J['local_time_int'] = O3J['time_local']
O3J['local_time_int'] = O3J['local_time_int'].str.replace(':00', '')
O3J['local_time_int'] = pd.to_numeric(O3J['local_time_int'])

# start changing the values
O3J.loc[O3J['local_time_int'] < 6, 'averageTra'] *= 0.1
O3J.loc[(O3J['local_time_int'] >= 6) & (O3J['local_time_int'] < 10), 'averageTra'] *= 2
O3J.loc[(O3J['local_time_int'] >= 10) & (O3J['local_time_int'] < 15), 'averageTra'] *= 1
O3J.loc[(O3J['local_time_int'] >= 15) & (O3J['local_time_int'] < 19), 'averageTra'] *= 2
O3J.loc[(O3J['local_time_int'] >= 19) & (O3J['local_time_int'] < 23), 'averageTra'] *= 0.5
O3J.loc[O3J['local_time_int'] == 23, 'averageTra'] *= 0.4

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

# add column for temp, wind dir, wind speed
O3J['temp'] = np.zeros(len(O3J))
O3J['wind_speed'] = np.zeros(len(O3J))
O3J['wind_dir'] = np.zeros(len(O3J))

# combine date and time into one column, convert it to datetime
for i in range(len(O3J['date_local'])):
    O3J['date_local'][i] = O3J['date_local'][i].replace('0:00:00', O3J['time_local'][i])
O3J['date_local'] = pd.to_datetime(O3J['date_local'], format='%m/%d/%Y %H:%M')

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
    except:
        newO3J.drop([index], inplace=True)
newO3J['HourlyDryBulbTemperature'] = pd.to_numeric(newO3J['HourlyDryBulbTemperature'])

O3TempJuly = newO3J.drop(axis=1, labels=['state_code', 'county_cod', 'site_numbe', 'parameter_', 'poc','datum', 'parameter','date_gmt', 'time_gmt','units_of_m',
       'units_of_1', 'sample_dur', 'sample_d_1', 'sample_fre', 'detection_','method_typ', 'method', 'method_cod', 'state', 'county', 'date_of_la','cbsa_code','Shape_Leng',
       'Shape_Area', 'Input_FID','local_time_int', 'temp', 'wind_speed', 'wind_dir','DATE', 'LATITUDE', 'LONGITUDE','REPORT_TYPE', 'SOURCE', 'HourlyAltimeterSetting'])

import matplotlib.pyplot as plt

plt.scatter(O3TempJuly['averageTra'],O3TempJuly['sample_mea'])
plt.show()

traffic = np.array(O3TempJuly['averageTra'])
traffic = traffic.reshape((-1,1))
temp = np.array(O3TempJuly['HourlyDryBulbTemperature']).reshape((-1,1))
oSamples = O3TempJuly['sample_mea']

model = LinearRegression()
model.fit(temp, oSamples)

r_sq = model.score(temp, oSamples)
print(f"coefficient of determination: {r_sq}")