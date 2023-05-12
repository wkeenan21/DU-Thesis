
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM,Dense

from tf.keras import layers, optimizers, losses, metrics, Model
from tf.keras import regularizers
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
Input(shape=)
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


