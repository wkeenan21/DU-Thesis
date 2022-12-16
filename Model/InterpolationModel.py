##Graph embedding
##attributed embedding for prediction
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, RepeatVector, Flatten, LSTM, Conv1D, Dense, Masking, Concatenate, Multiply, \
    AveragePooling1D, AveragePooling2D, AveragePooling3D, Permute, Reshape, \
    Add, MaxPooling2D, Dropout, Conv2D, Conv1D, Maximum, Average, MaxPooling1D, \
    MaxPooling3D, ConvLSTM2D
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras import regularizers
import numpy.ma as ma
import time
from tensorflow.keras import activations
import keras
import keras.backend as K


##todo: use LSTM + GraphSage(Local Aggr)
##todo: check masking for all

class HybridModel:
    ##todo: hours -- abs from the peak??
    ##two levels for two types of nodes: level 1 is finer, level 2 is coarser
    def __init__(self,  timesize_l1=3, timesize_l2=6, embed_dim=64, gcn_dim=64, batch_size=32, epo=100):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.fields =["LMA",	"Real"]
        self.inputdim_self = 1
        self.outputdim = 1

        ##the number of time steps used for analysis
        self.timesize_l1 = timesize_l1
        self.timesize_l2 = timesize_l2

        self.embed_dim = embed_dim
        self.gcn_dim = gcn_dim

        ##for graphsage
        self.batch_size = batch_size

        self.initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        self.l1_l2_reg = tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001)
        ##todo: add regularizer; replace optmizer with sdg?
        self.l1_reg = keras.regularizers.l1(0.001)
        self.l2_reg = keras.regularizers.l2(0.001)
        ##todo: readl name in data

        ##todo: most are hourly based, only predicted has a different time difference
        self.timediff_pred = 0
        self.timediff_normal = 1

        self.epo = epo
        self.powerratio = 8
        self.leveloption = 2  ##use level 2 only,  1 is both level
        self.gaphours = 0
        self.predcnt = 0
        self.traincnt = 0

        self.loss = losses.mae
        self.dropoutrate = 0.1
        self.useConv = 0


    ##todo: prepare data, stationinfo contains complete data; need to shift time dim
    def ResetDataTrainning(self, selfstationsids, self_timesteps, stationinfo_l2):
        ##node, time step,features, progressive moving
        self.node_cnt = len(selfstationsids)
        ##a list of sites for analysis
        self.stationids = selfstationsids
        self.first_hour = self.timesize_l1 * 24 + self.timesize_l2 + self.gaphours  ##todo: can remove timesize l2
        ##should be prediction rounds
        self.use_timesteps = self_timesteps - self.first_hour

        min_obs = 9999
        max_obs = -9999

        self.selfoutput_data = np.ones(self.use_timesteps * self.node_cnt * self.outputdim).reshape(self.use_timesteps,
                                                                                                    self.node_cnt,
                                                                                                    self.outputdim) * -9999.0
        self.self_lstm = np.ones(
            self.use_timesteps * self.node_cnt * self.timesize_l2 * self.inputdim_self).reshape(
            self.use_timesteps, self.node_cnt, self.timesize_l2, self.inputdim_self) * -9999.0

        self.selfoutput_data = ma.masked_where(self.selfoutput_data == -9999.0, self.selfoutput_data)
        self.self_lstm = ma.masked_where(self.self_lstm == -9999.0, self.self_lstm)

        ##self data--- shift by one more hour?
        pt_cnt = 0
        startpos = self.first_hour
        for pt1_id in selfstationsids:
            selfdata = stationinfo_l2[pt1_id][self.fields].to_numpy()
            selfdata = ma.masked_where(selfdata == -9999, selfdata)
            min_per = np.min(selfdata)
            max_per = np.max(selfdata)
            min_obs= min(min_obs,min_per)
            max_obs = max(max_per, max_obs)
            if np.isnan(selfdata[startpos:, :-1]).sum() > 0:
                print("Check")
            self.selfoutput_data[:, pt_cnt, :] = selfdata[startpos:, -1].reshape(self.use_timesteps, self.outputdim)
            pt_cnt = pt_cnt + 1

        pt_cnt = 0
        for pt1_id in selfstationsids:
            selfdata = stationinfo_l2[pt1_id][self.fields].to_numpy()
            selfdata = ma.masked_where(selfdata == -9999, selfdata)
            min_per = np.min(selfdata)
            max_per = np.max(selfdata)
            min_obs = min(min_obs, min_per)
            max_obs = max(max_per, max_obs)

            for i_timestep in range(self.use_timesteps):
                currenthour = self.first_hour + i_timestep
                hours_start = currenthour - self.timesize_l2 - self.gaphours
                for timesize in range(self.timesize_l2):  ##if gap not conisdered, longer time
                    hour_tmp = hours_start + timesize
                    self.self_lstm[i_timestep, pt_cnt, timesize, :] = selfdata[hour_tmp,0]
            pt_cnt = pt_cnt + 1


        self.selfoutput_data = (self.selfoutput_data - min_obs) / (max_obs - min_obs)
        self.self_lstm = (self.self_lstm - min_obs) / (max_obs - min_obs)
        self.max_value = max_obs
        self.min_value = min_obs

        ##todo: try [-1, 1] for variance

    def TrainModelProgressive(self, hours_used=480, traintype=0, recentsampleweight=1, waitround=3):
        totalsample = self.use_timesteps * self.traincnt
        total_rounds = int(totalsample / self.batch_size)  ##batch size should be factors of node_cnt
        print("total rounds: " + str(total_rounds))
        real_hour_per_batch = int(self.batch_size / self.traincnt)
        print("hours / round-batch: " + str(real_hour_per_batch))
        trainrounds = int(hours_used / real_hour_per_batch)
        print("train round: " + str(trainrounds))

        ##todo: recover prediction results
        self.stationids_predict = {}
        self.stationids_real = {}
        ##todo: first hour for the prediction
        self.first_hour_pred = trainrounds * real_hour_per_batch + self.first_hour  ##should be the idx not the count, start idx

        for i_node in range(self.node_cnt):
            site_id = self.stationids[i_node]
            self.stationids_predict[site_id] = []
            self.stationids_real[site_id] = []

        selfoutput_data = self.selfoutput_data[: hours_used, :self.traincnt]
        selfoutput_data = selfoutput_data.reshape(selfoutput_data.shape[0] * selfoutput_data.shape[1],
                                                  selfoutput_data.shape[2])
        selfinput_lstm_data = self.self_lstm[: hours_used, :self.traincnt]
        selfinput_lstm_data = selfinput_lstm_data.reshape(selfinput_lstm_data.shape[0] * selfinput_lstm_data.shape[1],
                                                          selfinput_lstm_data.shape[2], selfinput_lstm_data.shape[3])


        self.gnn_lstm_model.fit(
            x=[selfinput_lstm_data],
            y=selfoutput_data, batch_size=self.batch_size, epochs=self.epo, verbose=0)

        print("Rounds Left-----------------------: " + str(total_rounds - trainrounds - 1))
        startbatch_tmp = hours_used
        for i_round in range(total_rounds - trainrounds - 1):
            ##prediction todo: wait after certain rounds?

            selfoutput_data_pred = self.selfoutput_data[hours_used + i_round * real_hour_per_batch: hours_used + (
                        i_round + 1) * real_hour_per_batch]
            selfoutput_data_pred = selfoutput_data_pred.reshape(
                selfoutput_data_pred.shape[0] * selfoutput_data_pred.shape[1],
                selfoutput_data_pred.shape[2], )

            selfinput_lstm_data_pred = self.self_lstm[hours_used + i_round * real_hour_per_batch: hours_used + (
                    i_round + 1) * real_hour_per_batch]
            selfinput_lstm_data_pred = selfinput_lstm_data_pred.reshape(
                selfinput_lstm_data_pred.shape[0] * selfinput_lstm_data_pred.shape[1],
                selfinput_lstm_data_pred.shape[2], selfinput_lstm_data_pred.shape[3])

            selfoutput_data_pred_model = self.gnn_lstm_model.predict(   [ selfinput_lstm_data_pred,])

            ##todo: complete reset and use all previous data ; update with new data;  add more weights on old data

            if traintype == 1 or traintype == 2:  ##use old and new data
                print("res")
            elif type == 3:  ##only use new data to update every round
                ##refit/update the model
                # self.gnn_lstm_model.set_weights(self.modelweights)
                opt = keras.optimizers.Adam(learning_rate=0.005)
                self.gnn_lstm_model.compile(loss=self.loss, optimizer=opt, metrics=['accuracy'])

                self.gnn_lstm_model.fit(
                    x=[ selfinput_lstm_data_pred],
                    y=selfoutput_data_pred, batch_size=self.batch_size, epochs=self.epo, verbose=0)

            ##todo: recover data for saving
            selfoutput_data_pred_model[selfoutput_data_pred_model < 0] = 0

            selfoutput_data_pred = selfoutput_data_pred * (self.max_value - self.min_value) + self.min_value
            selfoutput_data_pred_model = selfoutput_data_pred_model * (
                        self.max_value - self.min_value) + self.min_value

            pred_hour = int(selfoutput_data_pred_model.shape[0] / self.node_cnt)

            for i_node in range(self.node_cnt):
                site_id = self.stationids[i_node]
                for i_pred_round in range(pred_hour):
                    self.stationids_predict[site_id].append(
                        selfoutput_data_pred_model[i_pred_round * self.node_cnt + i_node, 0])
                    ##store time series for every station
                    self.stationids_real[site_id].append(selfoutput_data_pred[i_pred_round * self.node_cnt + i_node, 0])

        cnt_site_all = 0
        totaldiff = 0
        self.total_hour_pred = len(self.stationids_real[site_id])
        for i_node in range(self.node_cnt):
            site_id = self.stationids[i_node]
            for i_hour in range(self.total_hour_pred):
                self.stationids_predict[site_id][i_hour] = ma.masked_where(
                    self.stationids_predict[site_id][i_hour] == -9999, self.stationids_predict[site_id][i_hour])
                self.stationids_real[site_id][i_hour] = ma.masked_where(
                    self.stationids_real[site_id][i_hour] == -9999, self.stationids_real[site_id][i_hour])
                diffarray1 = self.stationids_predict[site_id][i_hour] - self.stationids_real[site_id][i_hour]
                if ma.count_masked(diffarray1) > 0:
                    continue
                totaldiff = totaldiff + abs(diffarray1)
                cnt_site_all = cnt_site_all + 1

        return totaldiff / cnt_site_all, 0

    ##todo: repeat the prediction for all time step -->can leverage historical information from neighbors
    ##lstm for each category, ensemble the last step only
    def CreateLocalModelTimeSeries(self, featuredim=1, useVar=True):
        ##self node has the complete information
        nodeself = Input(shape=(self.timesize_l2, self.inputdim_self))
        lstmlayer = LSTM(self.gcn_dim)(nodeself)
        aq_final = Dense(self.outputdim)(lstmlayer)
        aq_final = Dropout(self.dropoutrate)(aq_final)
        gnn_lstm_model = Model(inputs=[nodeself], outputs=aq_final)
        gnn_lstm_model.compile(loss=self.loss, optimizer="adam", metrics=['accuracy'])
        self.gnn_lstm_model = gnn_lstm_model
        # dot_img_file = 'model_1.png'
        # tf.keras.utils.plot_model(gnn_lstm_model, to_file="Img/modelfilev_Sep_weatherlstm.png", show_shapes=True)

