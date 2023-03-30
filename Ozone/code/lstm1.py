import requests
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
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Import data
O3J = pd.read_csv(r"D:\Will_Git\DU-Thesis\Ozone\csvs\today\ozoneJulyDenver.csv")

# drop columns, convert to datetime
O3J['date_local'] = pd.to_datetime(O3J['date_local'])
O3J = O3J.drop(axis=1, labels=['Unnamed: 0.1', 'Unnamed: 0','uncertaint', 'qualifier', 'LULC','clipDEM1km','LATITUDE_1', 'LONGITUD_1', 'ELEVATION_x','NAME_x','ELEVATION_y', 'NAME_y','HourlyDewPointTemperature','HourlyPrecipitation',
                                      'HourlyPresentWeatherType','HourlyPressureChange', 'HourlyPressureTendency','HourlySkyConditions','HourlySeaLevelPressure', 'HourlyStationPressure', 'HourlyVisibility', 'HourlyWindDirection','HourlyWindGustSpeed','WindEquipmentChangeDate'])

# make wind speed numeric data and drop some wierd records
for index, row in O3J.iterrows():
    try:
        int(row['HourlyWindSpeed'])
    except:
        O3J.drop([index], inplace=True)
O3J['HourlyWindSpeed'] = pd.to_numeric(O3J['HourlyWindSpeed'])

# drop data points with close to zero ozone, they must be wrong
for index, row in O3J.iterrows():
    if row['sample_mea'] < 0.005:
        O3J.drop([index], inplace=True)


'''configure the model'''
timesize = 6 ##e.g., past 6 hours
input_var_cnt = 4 ##the number of variables used to perform prediction e.g., NO2, Ozone ... from the previous x tine steps
input_lstm = Input(shape=(timesize, input_var_cnt)) ##what is the input for every sample, sample count is not included every sample should be a 2D matrix
##prepare a LSTM layer
unit_lstm = 32 ##hidden dimensions transfer data, larger more complex model
lstmlayer = LSTM(unit_lstm) (input_lstm) ##this outputs a matrix of 1*unit_lstm, the format is the layer (input), the output of the layer stores the time series info and the interaction of variables..
denselayer = Dense(1)(lstmlayer) ## reduce the hidden dimension to 1 ==== output data ,1 value for 1 input sample --- predicted ozone
model = Model(inputs = input_lstm, outputs = denselayer)
model.compile(loss='mean_squared_error', optimizer='adam') ##how to measure the accuracy  compute mean squared error using all y_pred, y_true

# Loop through stations, run LSTM on data from each one, run regression on the results comparing actual vs expected value
stations = []
outputs = []
trains = []
for station in O3J['site_will'].unique():
    oneStation = O3J[O3J['site_will'] == station]
    oneStation.reset_index(inplace=True)
    stations.append(station)

    inputArray = np.zeros(shape=((len(oneStation)),6,4))
    outputArray = np.zeros(shape=((len(oneStation))))
    for index, row in oneStation.iterrows():
        if index < (len(oneStation) - 6):
            input = oneStation[['sample_mea','HourlyRelativeHumidity', 'HourlyDryBulbTemperature', 'HourlyWindSpeed']][index:index+6].to_numpy()
            inputArray[index] = input
            outputArray[index] = oneStation['sample_mea'][index+6]

    trainX = inputArray
    trainY = outputArray
    model.fit(trainX, trainY, epochs=100, batch_size=32, verbose=2)
    trainPredict = model.predict(trainX)

    outputs.append(outputArray)
    trains.append(trainPredict)

def runRegression(xvars, y):
    X = xvars.reshape(-1,1)
    y = y
    model1 = LinearRegression()
    model1.fit(X, y)

    r_sq = model1.score(X, y)
    print(f"coefficient of determination: {r_sq}")
    #print('coefficients '+ str(model1.coef_))
    #print('intercept '+ str(model1.intercept_))

for i in range(len(outputs)):
    print(stations[i])
    runRegression(xvars=outputs[i], y=trains[i])
