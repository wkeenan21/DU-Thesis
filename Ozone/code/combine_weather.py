import pandas as pd
import numpy as np
import os
import datetime as dt
#from PIL import Image
#from osgeo import gdal
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


O3J = pd.read_csv(r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\csvs\today\JulyPlus.csv")

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
O3J['day_of_week'] = O3J['date_local'].dt.day_name()

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
       'units_of_1', 'OID_', 'Join_Count', 'TARGET_FID','JOIN_FID','OID1','Field1','sample_dur', 'sample_d_1', 'sample_fre', 'detection_','method_typ', 'method', 'method_cod', 'state', 'county', 'date_of_la','cbsa_code','Shape_Leng',
       'Shape_Area', 'uncertaint','qualifier','Input_FID','local_time_int', 'temp', 'wind_speed', 'wind_dir','DATE', 'LATITUDE_1', 'LONGITUD_1','REPORT_TYPE', 'SOURCE', 'HourlyAltimeterSetting'])


O3TempJuly.to_csv(r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\csvs\today\JulyPlusWeather3.csv")
O3TempJuly = pd.read_csv(r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\csvs\today\JulyPlusWeather3.csv")
# phase shift traffic
justNoon = O3TempJuly.copy(deep=True)
phaseShift = O3TempJuly.copy(deep=True)
phaseShift.drop([0], inplace=True)
phaseShift.drop([1], inplace=True)
phaseShift.reset_index(drop=True, inplace=True)
phaseShift['shifted_tra'] = O3TempJuly['averageTra'][0:7856]
phaseShift = phaseShift[phaseShift['shifted_tra'].notna()]

for index, row in justNoon.iterrows():
    if row['time_local'] != '12:00' or row['sample_mea'] < 0.005:
        justNoon.drop([index], inplace=True)

import matplotlib.pyplot as plt
plt.scatter(justNoon['averageTra'],justNoon['sample_mea'])
plt.show()

X = phaseShift[['shifted_tra']]
y = phaseShift['sample_mea']

model = LinearRegression()
model.fit(X, y)

r_sq = model.score(X, y)
print(f"coefficient of determination: {r_sq}")

oneStation = O3TempJuly[O3TempJuly['site_will'] == 0]

inputArray = np.zeros(shape=(800,6,3))
outputArray = np.zeros(shape=(800))
for index, row in oneStation.iterrows():
    input = oneStation[['sample_mea','averageTra', 'HourlyDryBulbTemperature']][index:index+6].to_numpy()
    inputArray[index] = input
    outputArray[index] = oneStation['sample_mea'][index+6]


outputArray= outputArray.reshape(800,1)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM,Dense

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras import regularizers
import numpy.ma as ma
import time
from tensorflow.keras import activations
import keras
import keras.backend as K

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##todo: an example of LSTM model
# model = Model()
timesize = 6 ##e.g., past 6 hours
input_var_cnt = 3 ##the number of variables used to perform prediction e.g., NO2, Ozone ... from the previous x tine steps
input_lstm = Input(shape=(timesize, input_var_cnt)) ##what is the input for every sample, sample count is not included every sample should be a 2D matrix
##prepare a LSTM layer
unit_lstm = 32 ##hidden dimensions transfer data, larger more complex model
lstmlayer = LSTM(unit_lstm) (input_lstm) ##this outputs a matrix of 1*unit_lstm, the format is the layer (input), the output of the layer stores the time series info and the interaction of variables..
denselayer = Dense(1)(lstmlayer) ## reduce the hidden dimension to 1 ==== output data ,1 value for 1 input sample --- predicted ozone

##todo: configure the model
model = Model(inputs = input_lstm, outputs =  denselayer)
model.compile(loss='mean_squared_error', optimizer='adam') ##how to measure the accuracy  compute mean squared error using all y_pred, y_true

##todo: data preparation --replace with csv reading

trainX = inputArray
trainY = outputArray

# trainX = np.arange(100 * timesize * input_var_cnt).reshape(100, timesize, input_var_cnt ) ##input data for trainning, 100 is the sample cnt, should from the csv file
# trainY = np.arange(100) ##output data for trainning, 100 future ozone values correpsonding to the samples ...

# train model
model.fit(trainX, trainY, epochs=100, batch_size=32, verbose=2)
##make predictions or test model.
#testX = np.arange(100 * timesize * input_var_cnt).reshape(100, timesize, input_var_cnt )
trainPredict = model.predict(trainX)