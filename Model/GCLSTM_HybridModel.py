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
    def __init__(self, inputdim_self=11, neigbhorsize_l1=4, neighborsize_l2=4, outputdim=1,
                 timesize_l1=3, timesize_l2=6, embed_dim=64, gcn_dim=64, batch_size=32, epo=100, fields=[]):

        # self.fields_l1 = ["GHSValue", "DEMValue", "NLCD_recla", "PopDens", "Dist_Highw", "Dist_Major", "Traffic"]
        # self.fields_l2 = ["GHS", "DEM",  "NLCD_recla", "PopDens", "Dist_Highw", "Dist_Major","Traffic"]
        #
        # self.fields_l1 = ["DEMValue", "NLCD_recla", "PopDens", "Dist_Highw", "Dist_Major", "Traffic"]
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        if len(fields) > 0:
            self.fields_l2 = ["DEM", "NLCD_Recla", "PopDens", "Dist_HIghw", "Dist_Major", "Traffic",
                              "HourlyPrecipitation", "HourlyVisibility", "HourlyRelativeHumidity", "HourlyWindSpeed",
                              "HourlyDryBulbTemperature"]

        else:
            self.fields_l2 = ["DEM", "NLCD_recla", "PopDens", "Dist_Highw", "Dist_Major", "Traffic",
                              "HourlyPrecipitation", "HourlyVisibility", "HourlyRelativeHumidity", "HourlyWindSpeed",
                              "HourlyDryBulbTemperature"]

        self.fields_weather = ["HourlyPrecipitation", "HourlyVisibility", "HourlyRelativeHumidity", "HourlyWindSpeed",
                               "HourlyDryBulbTemperature"]
        self.fields_var = ["VarPM25", "VarPM25R"]
        self.fields_var_idx = 0
        self.inputdim_self = inputdim_self
        self.outputdim = outputdim

        ##the max neigbhoring features considered based on VDiagram
        self.neigbhorsize_l1 = neigbhorsize_l1
        self.neigbhorsize_l2 = neighborsize_l2
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
        self.useConv = 1 ##or 2

    ##todo: prepare data, stationinfo contains complete data; need to shift time dim
    # valid_epa_station_info, valid_lma_station_info, l1_predict_neighbor_ids, l1_predict_neighbor_dis, l1_train_neighbor_ids, l1_train_neighbor_dis,
    # l2_predict_neighbor_ids, l2_predict_neighbor_dis, l2_train_neighbor_ids, l2_train_neighbor_dis
    def ResetDataTrainning(self, selfstationsids, self_timesteps, stationinfo_l1, stationinfo_l2, neighborlist_l1_time,
                           l1_train_neighbor_dis_time,
                           neighborlist_l2_time, l2_train_neighbor_dis_time, pollutants="PM25", expdis=True,
                           logScale=False, negpos = True, rs="AOD"):
        ##node, time step,features, progressive moving
        self.node_cnt = len(selfstationsids)
        ##a list of sites for analysis
        self.stationids = selfstationsids
        self.first_hour = self.timesize_l1 * 24 + self.timesize_l2 + self.gaphours  ##todo: can remove timesize l2
        ##should be prediction rounds
        self.use_timesteps = self_timesteps - self.first_hour

        min_obs = 9999
        max_obs = -9999
        min_obs_Var = 9999
        max_obs_Var = -9999
        min_dis = 9999
        max_dis = -9999
        min_self = np.ones(self.inputdim_self + 1) * 9999
        max_self = np.ones(self.inputdim_self + 1) * -9999
        min_self_weather = np.ones(len(self.fields_weather)) * 9999
        max_self_weather = np.ones(len(self.fields_weather)) * -9999

        self.selfinputput_data = np.ones(self.use_timesteps * self.node_cnt * self.inputdim_self).reshape(
            self.use_timesteps, self.node_cnt, self.inputdim_self) * -9999.0
        self.selfoutput_data = np.ones(self.use_timesteps * self.node_cnt * self.outputdim).reshape(self.use_timesteps,
                                                                                                    self.node_cnt,
                                                                                                    self.outputdim) * -9999.0
        self.self_lstm_rs = np.ones(
            self.use_timesteps * self.node_cnt * self.timesize_l1 * 1).reshape(
            self.use_timesteps, self.node_cnt, self.timesize_l1, 1) * -9999.0

        self.self_lstm_weather = np.ones(
            self.use_timesteps * self.node_cnt * self.timesize_l2 * len(self.fields_weather)).reshape(
            self.use_timesteps, self.node_cnt, self.timesize_l2, len(self.fields_weather)) * -9999.0

        ##because l1 is reused, need to duplicate todo reverse distance
        self.neighborinput_data_l1 = np.ones(
            self.use_timesteps * self.node_cnt * self.neigbhorsize_l1 * self.timesize_l1 * 2). \
                                         reshape(self.use_timesteps, self.node_cnt, self.timesize_l1,
                                                 self.neigbhorsize_l1, 2) * -9999.0
        ##todo: the real time size for l2 can be self.timesize_l2 +  self.timesize_l1*24
        self.neighborinput_data_l2 = np.ones(
            self.use_timesteps * self.node_cnt * self.neigbhorsize_l2 * self.timesize_l2 * 2). \
                                         reshape(self.use_timesteps, self.node_cnt, self.timesize_l2,
                                                 self.neigbhorsize_l2, 2) * -9999.0
        self.neighborinput_data_l2_Var = np.ones(
            self.use_timesteps * self.node_cnt * self.neigbhorsize_l2 * self.timesize_l2 * 1). \
                                             reshape(self.use_timesteps, self.node_cnt, self.timesize_l2,
                                                     self.neigbhorsize_l2, 1) * -9999.0

        self.selfinputput_data = ma.masked_where(self.selfinputput_data == -9999.0, self.selfinputput_data)
        self.neighborinput_data_l1 = ma.masked_where(self.neighborinput_data_l1 == -9999.0, self.neighborinput_data_l1)
        self.neighborinput_data_l2 = ma.masked_where(self.neighborinput_data_l2 == -9999.0, self.neighborinput_data_l2)
        self.neighborinput_data_l2_Var = ma.masked_where(self.neighborinput_data_l2_Var == -9999.0,
                                                         self.neighborinput_data_l2_Var)
        self.selfoutput_data = ma.masked_where(self.selfoutput_data == -9999.0, self.selfoutput_data)
        self.self_lstm_weather = ma.masked_where(self.self_lstm_weather == -9999.0, self.self_lstm_weather)
        self.self_lstm_rs = ma.masked_where(self.self_lstm_rs == -9999.0, self.self_lstm_rs)

        ##self data--- shift by one more hour?
        pt_cnt = 0
        startpos = self.first_hour
        for pt1_id in selfstationsids:
            selfdata = stationinfo_l2[pt1_id][self.fields_l2 + [pollutants]].to_numpy()
            selfdata = ma.masked_where(selfdata == -9999, selfdata)
            min_per = np.min(selfdata, axis=0)
            max_per = np.max(selfdata, axis=0)
            for i in range(self.inputdim_self + 1):
                min_self[i] = min(min_per[i], min_self[i])
                max_self[i] = max(max_per[i], max_self[i])
            if np.isnan(selfdata[startpos:, :-1]).sum() > 0:
                print("Check")
            self.selfinputput_data[:, pt_cnt, :] = selfdata[startpos:, :-1].reshape(self.use_timesteps,
                                                                                    self.inputdim_self)
            self.selfoutput_data[:, pt_cnt, :] = selfdata[startpos:, -1].reshape(self.use_timesteps, self.outputdim)
            pt_cnt = pt_cnt + 1

        ##todo: rs data
        pt_cnt = 0
        for pt1_id in selfstationsids:
            selfdata_rs = stationinfo_l2[pt1_id][rs].to_numpy()
            selfdata_rs = ma.masked_where(selfdata_rs == -9999, selfdata_rs)
            for i_timestep in range(self.use_timesteps):
                currenthour = self.first_hour + i_timestep
                days_start = int((currenthour - self.gaphours) / 24) - self.timesize_l1
                for timesize in range(self.timesize_l1):
                    ##every day has the same value?
                    hour_tmp = (days_start + timesize) * 24  ##middle day obs-- 24 hours same
                    self.self_lstm_rs[i_timestep, pt_cnt, timesize, :] = ma.average(
                        selfdata_rs[hour_tmp: hour_tmp + 24], axis=0)
            pt_cnt = pt_cnt + 1

        # print(np.argwhere(np.isnan(self.selfinputput_data)))
        pt_cnt = 0
        for pt1_id in selfstationsids:
            selfdata_weather = stationinfo_l2[pt1_id][self.fields_weather].to_numpy()
            selfdata_weather = ma.masked_where(selfdata_weather == -9999, selfdata_weather)
            min_per_weather = np.min(selfdata_weather, axis=0)
            max_per_weather = np.max(selfdata_weather, axis=0)

            for i in range(len(self.fields_weather)):
                min_self_weather[i] = min(min_per_weather[i], min_self_weather[i])
                max_self_weather[i] = max(max_per_weather[i], max_self_weather[i])
            for i_timestep in range(self.use_timesteps):
                currenthour = self.first_hour + i_timestep
                hours_start = currenthour - self.timesize_l2 - self.gaphours
                for timesize in range(self.timesize_l2):  ##if gap not conisdered, longer time
                    hour_tmp = hours_start + timesize
                    self.self_lstm_weather[i_timestep, pt_cnt, timesize, :] = selfdata_weather[hour_tmp]
            pt_cnt = pt_cnt + 1

        ##level 1: epa data
        pt_cnt = 0
        for pt1_id in selfstationsids:
            for timestep in range(self.use_timesteps):
                currenthour = self.first_hour + timestep
                pt2_id_list = neighborlist_l1_time[currenthour][pt1_id]
                pt2_dis_list = l1_train_neighbor_dis_time[currenthour][pt1_id]
                ##select the data frame for a station
                neighbor_node_cnt_per_l1 = 0
                for pt2_id in pt2_id_list:
                    ##access the data frame
                    neighbor_array_time = stationinfo_l1[pt2_id][pollutants].to_numpy()
                    neighbor_array_distance = pt2_dis_list[neighbor_node_cnt_per_l1]
                    min_dis = min(min_dis, neighbor_array_distance)
                    max_dis = max(max_dis, neighbor_array_distance)
                    min_obs = min(min_obs, np.min(neighbor_array_time))
                    max_obs = max(max_obs, np.max(neighbor_array_time))
                    days_start = int((currenthour - self.gaphours) / 24) - self.timesize_l1
                    # days_end = int(hours_real/24)
                    for timesize in range(self.timesize_l1):
                        ##every day has the same value?
                        hour_tmp = (days_start + timesize) * 24 + 12  ##middle day obs-- 24 hours same
                        self.neighborinput_data_l1[timestep, pt_cnt, timesize, neighbor_node_cnt_per_l1, 0] = \
                            neighbor_array_time[hour_tmp]
                        self.neighborinput_data_l1[
                            timestep, pt_cnt, timesize, neighbor_node_cnt_per_l1, 1] = neighbor_array_distance
                    neighbor_node_cnt_per_l1 = neighbor_node_cnt_per_l1 + 1
                    if neighbor_node_cnt_per_l1 == self.neigbhorsize_l1:
                        break
            pt_cnt = pt_cnt + 1

        ##level 2: lma data
        pt_cnt = 0
        for pt1_id in selfstationsids:
            for timestep in range(self.use_timesteps):
                currenthour = self.first_hour + timestep
                pt2_id_list = neighborlist_l2_time[currenthour][pt1_id]
                pt2_dis_list = l2_train_neighbor_dis_time[currenthour][pt1_id]
                ##select the data frame for a station
                neighbor_node_cnt_per_l2 = 0
                for pt2_id in pt2_id_list:
                    ##access the data frame
                    neighbor_array_time = stationinfo_l2[pt2_id][pollutants].to_numpy()
                    neighbor_array_time_var = stationinfo_l2[pt2_id][self.fields_var[self.fields_var_idx]].to_numpy()
                    neighbor_array_distance = pt2_dis_list[neighbor_node_cnt_per_l2]
                    min_dis = min(min_dis, neighbor_array_distance)
                    max_dis = max(max_dis, neighbor_array_distance)
                    min_obs = min(min_obs, np.min(neighbor_array_time))
                    max_obs = max(max_obs, np.max(neighbor_array_time))
                    min_obs_Var = min(min_obs_Var, np.min(neighbor_array_time_var))
                    max_obs_Var = max(max_obs_Var, np.max(neighbor_array_time_var))

                    hours_start = currenthour - self.timesize_l2 - self.gaphours
                    # days_end = int(hours_real/24)
                    for timesize in range(self.timesize_l2):
                        ##every day has the same value?
                        hour_tmp = hours_start + timesize
                        self.neighborinput_data_l2[
                            timestep, pt_cnt, timesize, neighbor_node_cnt_per_l2, 0] = neighbor_array_time[hour_tmp]
                        self.neighborinput_data_l2_Var[
                            timestep, pt_cnt, timesize, neighbor_node_cnt_per_l2, 0] = neighbor_array_time_var[hour_tmp]
                        # if neighbor_array_time[hour_tmp].min()==-9999:
                        #     print("Wrong")
                        self.neighborinput_data_l2[
                            timestep, pt_cnt, timesize, neighbor_node_cnt_per_l2, 1] = neighbor_array_distance

                        # if neighbor_array_distance.min()==-9999:
                        #     print("Wrong")
                    neighbor_node_cnt_per_l2 = neighbor_node_cnt_per_l2 + 1
                    if neighbor_node_cnt_per_l2 == self.neigbhorsize_l2:
                        break
            pt_cnt = pt_cnt + 1

        print("Check Min--------------------")
        print(self.selfinputput_data.min())
        print(self.selfoutput_data.min())
        print(self.neighborinput_data_l1.min())
        print(self.neighborinput_data_l2.min())
        print(self.neighborinput_data_l2_Var.min())
        if self.neighborinput_data_l1.min() < 0 or self.neighborinput_data_l2.min() < 0:
            print("Wrong")
        print("Check Min Finish---------------------------------")

        ##todo: normalize data-- dis can be power-based
        self.neighborinput_data_l1[:, :, :, :, 0] = (self.neighborinput_data_l1[:, :, :, :, 0] - min_obs) / (
                max_obs - min_obs)
        self.neighborinput_data_l1[:, :, :, :, 1] = (self.neighborinput_data_l1[:, :, :, :, 1] - min_dis) / (
                max_dis - min_dis)
        self.neighborinput_data_l2[:, :, :, :, 0] = (self.neighborinput_data_l2[:, :, :, :, 0] - min_obs) / (
                max_obs - min_obs)

        ##Todo: -1, 1
        if negpos:
            abs_extre_obs_var = max(abs(min_obs_Var), abs(max_obs_Var))
            self.neighborinput_data_l2_Var[:, :, :, :, 0] =  self.neighborinput_data_l2_Var[:, :, :, :, 0] / abs_extre_obs_var
        else:
            self.neighborinput_data_l2_Var[:, :, :, :, 0] = (self.neighborinput_data_l2_Var[:, :, :, :,
                                                             0] - min_obs_Var) / (
                                                                    max_obs_Var - min_obs_Var)

        self.neighborinput_data_l2[:, :, :, :, 1] = (self.neighborinput_data_l2[:, :, :, :, 1] - min_dis) / (
                max_dis - min_dis)

        if logScale:
            self.selfoutput_data = np.log2(self.selfoutput_data + 1)
            self.log_min = np.min(self.selfoutput_data)
            self.log_max = np.max(self.selfoutput_data)
            self.selfoutput_data = (self.selfoutput_data - self.log_min) / (self.log_max - self.log_min)
            ## - min_self[-1]) / (max_self[-1] - min_self[-1])
        else:
            self.selfoutput_data = (self.selfoutput_data - min_self[-1]) / (max_self[-1] - min_self[-1])

        self.logScale = logScale

        self.selfinputput_data = (self.selfinputput_data - min_self[:-1]) / (max_self[:-1] - min_self[:-1])
        self.min_value = min_self[-1]
        self.max_value = max_self[-1]
        self.self_lstm_weather = (self.self_lstm_weather - min_self_weather) / (max_self_weather - min_self_weather)

        min_rs = np.nanmin(self.self_lstm_rs)
        max_rs = np.nanmax(self.self_lstm_rs.max())
        self.self_lstm_rs = (self.self_lstm_rs - min_rs) / (max_rs - min_rs)

        if expdis:
            self.neighborinput_data_l2[:, :, :, :, 1] = np.exp(
                np.power(self.neighborinput_data_l2[:, :, :, :, 1], 2) * -1)
            self.neighborinput_data_l1[:, :, :, :, 1] = np.exp(
                np.power(self.neighborinput_data_l1[:, :, :, :, 1], 2) * -1)

        print("Check Normalized Min--------------------")
        print(self.selfinputput_data.min())
        print(self.selfoutput_data.min())
        print(self.neighborinput_data_l1.min())
        print(self.neighborinput_data_l2.min())
        print(self.neighborinput_data_l2_Var.min())
        print(self.self_lstm_weather.min())
        print("Check Normalized Min Finish---------------------------------")

        ##todo: try [-1, 1] for variance
    def CheckValidIdx(self, selfinput_lstm_rs_data):
        valididx = np.zeros(selfinput_lstm_rs_data.shape[0])
        cnt = selfinput_lstm_rs_data.shape[0]
        for i_cnt in range(cnt):
            array_tmp = selfinput_lstm_rs_data[i_cnt]
            if ma.count_masked(array_tmp) == 0:
                valididx[i_cnt] = 1
        # valididx = np.array(valididx, dtype=bool)
        return valididx!=0

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
        self.stationids_predict_rs = {}
        self.stationids_predict_rs_used = {}
        ##todo: first hour for the prediction
        self.first_hour_pred = trainrounds * real_hour_per_batch + self.first_hour  ##should be the idx not the count, start idx

        for i_node in range(self.node_cnt):
            site_id = self.stationids[i_node]
            self.stationids_predict[site_id] = []
            self.stationids_real[site_id] = []
            self.stationids_predict_rs[site_id] = []
            self.stationids_predict_rs_used[site_id] = []

        neighborinput_data_l1 = self.neighborinput_data_l1[: hours_used, :self.traincnt]
        neighborinput_data_l2 = self.neighborinput_data_l2[: hours_used, : self.traincnt]
        neighborinput_data_l2_Var = self.neighborinput_data_l2_Var[: hours_used, : self.traincnt]
        neighborinput_data_l1 = neighborinput_data_l1.reshape(
            neighborinput_data_l1.shape[0] * neighborinput_data_l1.shape[1], neighborinput_data_l1.shape[2],
            neighborinput_data_l1.shape[3], neighborinput_data_l1.shape[4])
        neighborinput_data_l2 = neighborinput_data_l2.reshape(
            neighborinput_data_l2.shape[0] * neighborinput_data_l2.shape[1], neighborinput_data_l2.shape[2],
            neighborinput_data_l2.shape[3], neighborinput_data_l2.shape[4])
        neighborinput_data_l2_Var = neighborinput_data_l2_Var.reshape(
            neighborinput_data_l2_Var.shape[0] * neighborinput_data_l2_Var.shape[1], neighborinput_data_l2_Var.shape[2],
            neighborinput_data_l2_Var.shape[3], neighborinput_data_l2_Var.shape[4])

        neighborinput_data_l1_feature = neighborinput_data_l1[:, :, :, 0].reshape(
            neighborinput_data_l1.shape[0], neighborinput_data_l1.shape[1], neighborinput_data_l1.shape[2], 1)
        neighborinput_data_l2_feature = neighborinput_data_l2[:, :, :, 0].reshape(
            neighborinput_data_l2.shape[0], neighborinput_data_l2.shape[1], neighborinput_data_l2.shape[2], 1)
        neighborinput_data_l2_feature_Var = neighborinput_data_l2_Var[:, :, :, 0].reshape(
            neighborinput_data_l2_Var.shape[0], neighborinput_data_l2_Var.shape[1], neighborinput_data_l2_Var.shape[2],
            1)

        neighborinput_data_l1_dis = neighborinput_data_l1[:, :, :, 1].reshape(
            neighborinput_data_l1.shape[0], neighborinput_data_l1.shape[1], neighborinput_data_l1.shape[2], 1)
        neighborinput_data_l2_dis = neighborinput_data_l2[:, :, :, 1].reshape(
            neighborinput_data_l2.shape[0], neighborinput_data_l2.shape[1], neighborinput_data_l2.shape[2], 1)

        selfoutput_data = self.selfoutput_data[: hours_used, :self.traincnt]
        selfoutput_data = selfoutput_data.reshape(selfoutput_data.shape[0] * selfoutput_data.shape[1],
                                                  selfoutput_data.shape[2])
        selfinputput_data = self.selfinputput_data[: hours_used, :self.traincnt]
        selfinputput_data = selfinputput_data.reshape(selfinputput_data.shape[0] * selfinputput_data.shape[1],
                                                      selfinputput_data.shape[2])
        selfinput_lstm_data = self.self_lstm_weather[: hours_used, :self.traincnt]
        selfinput_lstm_data = selfinput_lstm_data.reshape(selfinput_lstm_data.shape[0] * selfinput_lstm_data.shape[1],
                                                          selfinput_lstm_data.shape[2], selfinput_lstm_data.shape[3])

        selfinput_lstm_rs_data = self.self_lstm_rs[: hours_used, :self.traincnt]
        selfinput_lstm_rs_data = selfinput_lstm_rs_data.reshape(
            selfinput_lstm_rs_data.shape[0] * selfinput_lstm_rs_data.shape[1],
            selfinput_lstm_rs_data.shape[2], selfinput_lstm_rs_data.shape[3])

        ##skip invalid sites
        valid_day_list = self.CheckValidIdx(selfinput_lstm_rs_data)
        valid_cnt = np.sum(valid_day_list)

        if self.leveloption == 1:
            self.gnn_lstm_model.fit(
                x=[neighborinput_data_l1_feature,
                   neighborinput_data_l1_dis, neighborinput_data_l2_feature, neighborinput_data_l2_feature_Var,
                   neighborinput_data_l2_dis, selfinputput_data, selfinput_lstm_data],
                y=selfoutput_data, batch_size=self.batch_size, epochs=self.epo, verbose=0)

        else:
            self.gnn_lstm_model.fit(
                x=[neighborinput_data_l2_feature, neighborinput_data_l2_feature_Var,
                   neighborinput_data_l2_dis, selfinputput_data, selfinput_lstm_data],
                y=selfoutput_data, batch_size=self.batch_size, epochs=self.epo, verbose=0)



        print("Rounds Left-----------------------: " + str(total_rounds - trainrounds - 1))
        startbatch_tmp = hours_used
        for i_round in range(total_rounds - trainrounds - 1):
            ##prediction todo: wait after certain rounds?
            neighborinput_data_l1_pred = self.neighborinput_data_l1[
                                         hours_used + i_round * real_hour_per_batch: hours_used + (
                                                     i_round + 1) * real_hour_per_batch]
            neighborinput_data_l1_pred = neighborinput_data_l1_pred.reshape(
                neighborinput_data_l1_pred.shape[0] * neighborinput_data_l1_pred.shape[1],
                neighborinput_data_l1_pred.shape[2], neighborinput_data_l1_pred.shape[3],
                neighborinput_data_l1_pred.shape[4])
            neighborinput_data_l1_pred_feature = neighborinput_data_l1_pred[:, :, :, 0].reshape(
                neighborinput_data_l1_pred.shape[0], neighborinput_data_l1_pred.shape[1],
                neighborinput_data_l1_pred.shape[2], 1)
            neighborinput_data_l1_pred_dis = neighborinput_data_l1_pred[:, :, :, 1].reshape(
                neighborinput_data_l1_pred.shape[0], neighborinput_data_l1_pred.shape[1],
                neighborinput_data_l1_pred.shape[2], 1)

            neighborinput_data_l2_pred = self.neighborinput_data_l2[
                                         hours_used + i_round * real_hour_per_batch: hours_used + (
                                                     i_round + 1) * real_hour_per_batch]
            neighborinput_data_l2_pred = neighborinput_data_l2_pred.reshape(
                neighborinput_data_l2_pred.shape[0] * neighborinput_data_l2_pred.shape[1],
                neighborinput_data_l2_pred.shape[2], neighborinput_data_l2_pred.shape[3],
                neighborinput_data_l2_pred.shape[4])

            neighborinput_data_l2_pred_Var = self.neighborinput_data_l2_Var[
                                             hours_used + i_round * real_hour_per_batch: hours_used + (
                                                     i_round + 1) * real_hour_per_batch]
            neighborinput_data_l2_pred_Var = neighborinput_data_l2_pred_Var.reshape(
                neighborinput_data_l2_pred_Var.shape[0] * neighborinput_data_l2_pred_Var.shape[1],
                neighborinput_data_l2_pred_Var.shape[2], neighborinput_data_l2_pred_Var.shape[3],
                neighborinput_data_l2_pred_Var.shape[4])

            neighborinput_data_l2_pred_feature = neighborinput_data_l2_pred[:, :, :, 0].reshape(
                neighborinput_data_l2_pred.shape[0], neighborinput_data_l2_pred.shape[1],
                neighborinput_data_l2_pred.shape[2], 1)
            neighborinput_data_l2_pred_feature_Var = neighborinput_data_l2_pred_Var[:, :, :, 0].reshape(
                neighborinput_data_l2_pred_Var.shape[0], neighborinput_data_l2_pred_Var.shape[1],
                neighborinput_data_l2_pred_Var.shape[2], 1)

            neighborinput_data_l2_pred_dis = neighborinput_data_l2_pred[:, :, :, 1].reshape(
                neighborinput_data_l2_pred.shape[0], neighborinput_data_l2_pred.shape[1],
                neighborinput_data_l2_pred.shape[2], 1)

            selfoutput_data_pred = self.selfoutput_data[hours_used + i_round * real_hour_per_batch: hours_used + (
                        i_round + 1) * real_hour_per_batch]
            selfoutput_data_pred = selfoutput_data_pred.reshape(
                selfoutput_data_pred.shape[0] * selfoutput_data_pred.shape[1],
                selfoutput_data_pred.shape[2], )
            selfinputput_data_pred = self.selfinputput_data[hours_used + i_round * real_hour_per_batch: hours_used + (
                        i_round + 1) * real_hour_per_batch]

            selfinputput_data_pred = selfinputput_data_pred.reshape(
                selfinputput_data_pred.shape[0] * selfinputput_data_pred.shape[1],
                selfinputput_data_pred.shape[2], )

            selfinput_lstm_data_pred = self.self_lstm_weather[hours_used + i_round * real_hour_per_batch: hours_used + (
                        i_round + 1) * real_hour_per_batch]
            selfinput_lstm_data_pred = selfinput_lstm_data_pred.reshape(
                selfinput_lstm_data_pred.shape[0] * selfinput_lstm_data_pred.shape[1],
                selfinput_lstm_data_pred.shape[2], selfinput_lstm_data_pred.shape[3])

            selfinput_lstm_rs_data_pred = self.self_lstm_rs[hours_used + i_round * real_hour_per_batch: hours_used + (
                    i_round + 1) * real_hour_per_batch]
            selfinput_lstm_rs_data_pred = selfinput_lstm_rs_data_pred.reshape(
                selfinput_lstm_rs_data_pred.shape[0] * selfinput_lstm_rs_data_pred.shape[1],
                selfinput_lstm_rs_data_pred.shape[2], selfinput_lstm_rs_data_pred.shape[3])

            valid_day_list = self.CheckValidIdx(selfinput_lstm_rs_data_pred)
            valid_cnt = np.sum(valid_day_list)

            if self.leveloption == 1:
                selfoutput_data_pred_model = self.gnn_lstm_model.predict(
                    [neighborinput_data_l1_pred_feature, neighborinput_data_l1_pred_dis,
                     neighborinput_data_l2_pred_feature, neighborinput_data_l2_pred_feature_Var,
                     neighborinput_data_l2_pred_dis,
                     selfinputput_data_pred, selfinput_lstm_data_pred])
                selfoutput_data_pred_model_rs = selfoutput_data_pred_model.copy()
                selfoutput_data_pred_model_rs_used = valid_day_list

            else:
                selfoutput_data_pred_model = self.gnn_lstm_model.predict(
                    [neighborinput_data_l2_pred_feature, neighborinput_data_l2_pred_feature_Var,
                     neighborinput_data_l2_pred_dis,
                     selfinputput_data_pred, selfinput_lstm_data_pred])
            ##todo: complete reset and use all previous data ; update with new data;  add more weights on old data

                selfoutput_data_pred_model_rs = selfoutput_data_pred_model.copy()
                selfoutput_data_pred_model_rs_used = valid_day_list

            if traintype == 1 or traintype == 2:  ##use old and new data
                print("res")
            elif type == 3:  ##only use new data to update every round
                ##refit/update the model
                # self.gnn_lstm_model.set_weights(self.modelweights)
                opt = keras.optimizers.Adam(learning_rate=0.005)
                self.gnn_lstm_model.compile(loss=self.loss, optimizer=opt, metrics=['accuracy'])
                if self.leveloption == 1:
                    for i_fit_round in range(recentsampleweight):
                        self.gnn_lstm_model.fit(
                            x=[neighborinput_data_l1_pred_feature, neighborinput_data_l1_pred_dis,
                               neighborinput_data_l2_pred_feature, neighborinput_data_l2_pred_feature_Var,
                               neighborinput_data_l2_pred_dis, selfinputput_data_pred, selfinput_lstm_data_pred],
                            y=selfoutput_data_pred, batch_size=self.batch_size, epochs=self.epo, verbose=0)


                else:
                    for i_fit_round in range(recentsampleweight):
                        self.gnn_lstm_model.fit(
                            x=[neighborinput_data_l2_pred_feature, neighborinput_data_l2_pred_feature_Var,
                               neighborinput_data_l2_pred_dis, selfinputput_data_pred, selfinput_lstm_data_pred],
                            y=selfoutput_data_pred, batch_size=self.batch_size, epochs=self.epo, verbose=0)


            ##todo: recover data for saving
            selfoutput_data_pred_model[selfoutput_data_pred_model < 0] = 0
            selfoutput_data_pred_model_rs[selfoutput_data_pred_model_rs < 0] = 0
            if self.logScale:
                selfoutput_data_pred = np.power(2,
                                                selfoutput_data_pred * (self.log_max - self.log_min) + self.log_min) - 1
                selfoutput_data_pred_model = np.power(2, selfoutput_data_pred_model * (
                        self.log_max - self.log_min) + self.log_min) - 1
            else:
                selfoutput_data_pred = selfoutput_data_pred * (self.max_value - self.min_value) + self.min_value
                selfoutput_data_pred_model = selfoutput_data_pred_model * (
                            self.max_value - self.min_value) + self.min_value
                selfoutput_data_pred_model_rs = selfoutput_data_pred_model_rs * (
                        self.max_value - self.min_value) + self.min_value

            pred_hour = int(selfoutput_data_pred_model.shape[0] / self.node_cnt)

            for i_node in range(self.node_cnt):
                site_id = self.stationids[i_node]
                for i_pred_round in range(pred_hour):
                    self.stationids_predict[site_id].append(
                        selfoutput_data_pred_model[i_pred_round * self.node_cnt + i_node, 0])
                    self.stationids_predict_rs_used[site_id].append(
                        selfoutput_data_pred_model_rs_used[i_pred_round * self.node_cnt + i_node])
                    self.stationids_predict_rs[site_id].append(
                        selfoutput_data_pred_model_rs[i_pred_round * self.node_cnt + i_node, 0])
                    ##store time series for every station
                    self.stationids_real[site_id].append(selfoutput_data_pred[i_pred_round * self.node_cnt + i_node, 0])

        cnt_site_all = 0
        totaldiff = 0
        totaldiff_rs = 0
        self.total_hour_pred = len(self.stationids_real[site_id])
        for i_node in range(self.node_cnt):
            site_id = self.stationids[i_node]
            for i_hour in range(self.total_hour_pred):
                self.stationids_predict[site_id][i_hour] = ma.masked_where(
                    self.stationids_predict[site_id][i_hour] == -9999, self.stationids_predict[site_id][i_hour])
                self.stationids_real[site_id][i_hour] = ma.masked_where(
                    self.stationids_real[site_id][i_hour] == -9999, self.stationids_real[site_id][i_hour])
                self.stationids_predict_rs[site_id][i_hour] = ma.masked_where(
                    self.stationids_predict_rs[site_id][i_hour] == -9999, self.stationids_predict_rs[site_id][i_hour])
                diffarray1 = self.stationids_predict[site_id][i_hour] - self.stationids_real[site_id][i_hour]
                if ma.count_masked(diffarray1) > 0:
                    continue
                totaldiff = totaldiff + abs(diffarray1)
                diffarray2 = self.stationids_predict_rs[site_id][i_hour] - self.stationids_real[site_id][i_hour]
                totaldiff_rs = totaldiff_rs + abs(diffarray2)
                cnt_site_all = cnt_site_all + 1

        return totaldiff / cnt_site_all, totaldiff_rs / cnt_site_all




    ##todo: repeat the prediction for all time step -->can leverage historical information from neighbors
    ##lstm for each category, ensemble the last step only
    def CreateLocalModelTimeSeries(self, featuredim=1, useVar=True):
        ##self node has the complete information
        nodeself = Input(shape=(self.inputdim_self))


        nodeself_weather_lstm = Input(shape=(self.timesize_l2, len(self.fields_weather)))


        # kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4),
        # bias_regularizer = regularizers.l2(1e-4),
        # activity_regularizer = regularizers.l2(1e-5)

        neighbor_time_list_l2_feature = Input(shape=(self.timesize_l2, self.neigbhorsize_l2, featuredim))
        neighbor_time_list_l2_dis = Input(shape=(self.timesize_l2, self.neigbhorsize_l2, 1))



        aq_l2 = self.CreateLocalModelAggrDynamic(neighbor_time_list_l2_feature, neighbor_time_list_l2_dis,
                                                 self.neigbhorsize_l2, self.timesize_l2, )
        aq_l2 = Dropout(self.dropoutrate)(aq_l2)

        neighbor_time_list_l2_feature_Var = Input(shape=(self.timesize_l2, self.neigbhorsize_l2, featuredim))


        ##todo: how to consider time difference? e.g., hour 1 vs hour 12
        ##todo: scale the props first------perhaps no termporal info

        # nodeprops_self = LSTM(self.embed_dim)(nodepros_emb)
        ##todo: combine aq_l1, aq_l2, nodeprops
        ##todo: learn weights or attention scores--self attention? simple first

        aq_combine = aq_l2

        aq_final = Dense(self.outputdim, activation='tanh')(aq_combine)
        aq_final = Dropout(self.dropoutrate)(aq_final)


        gnn_lstm_model = Model(inputs=[
            neighbor_time_list_l2_feature, neighbor_time_list_l2_feature_Var, neighbor_time_list_l2_dis,
            nodeself, nodeself_weather_lstm], outputs=aq_final)

        gnn_lstm_model.compile(loss=self.loss, optimizer="adam", metrics=['accuracy'])
        self.gnn_lstm_model = gnn_lstm_model
        # dot_img_file = 'model_1.png'
        # tf.keras.utils.plot_model(gnn_lstm_model, to_file="Img/modelfilev_Sep_weatherlstm.png", show_shapes=True)


    ##todo: var case does not scale?
    def CreateLocalModelAggrDynamic(self, neighbor_list, neighbor_dis_list, neighborhoodsize, timedimension=1,
                                    useMax=True):
        ##todo: remove the dense
        # neighbor_scale = Dense(self.embed_dim, )(neighbor_list)
        neighbor_scale = Dropout(self.dropoutrate)(neighbor_list)
        # dis_scale = Dense(self.embed_dim, name="NeighborDisScale")(neighbor_dis_list)
        # negative_exp_power_dis = NegativePowerLayer()(dis_scale)
        # neigbhor_edge_weight = Multiply()([neighbor_scale, negative_exp_power_dis])
        neigbhor_edge_weight = Multiply()([neighbor_scale, neighbor_dis_list])
        ##todo: avg pooling, min, max, sum pooling or weighted average?--not working?
        ##todo: use Conv2D?


        if self.useConv == 1:
            # 1d "fake" time series aggregration
            avg_layer = Conv1D(filters=self.embed_dim, kernel_size=(neighborhoodsize,),
                               padding='valid', )(neigbhor_edge_weight)
            ##add a chanel
        elif self.useConv == 2:
            neigbhor_edge_weight_reshape = Reshape((timedimension, neighborhoodsize, self.embed_dim, 1))(
                neigbhor_edge_weight)
            filter_cnt = 4
            avg_conv2d = Conv2D(filters=filter_cnt, kernel_size=(neighborhoodsize, 1),
                                strides=(neighborhoodsize, 1),
                                padding='valid', )(neigbhor_edge_weight_reshape)
            ##average, max, or dense
            avg_conv2d = Dropout(self.dropoutrate)(avg_conv2d)
            avg_conv2d_combine = Dense(1)(avg_conv2d)
            # avg_conv2d_combine  = MaxPooling3D(pool_size=(1,1,filter_cnt))(avg_conv2d)
            # avg_conv2d_combine = AveragePooling3D(pool_size=(1,1, filter_cnt))(avg_conv2d)
            avg_layer = Reshape((timedimension, 1, self.embed_dim))(avg_conv2d_combine)
            avg_layer = Dropout(self.dropoutrate)(avg_layer)
            time_avg_layer = Reshape((timedimension, self.embed_dim))(avg_layer)
            aq_neighbor = LSTM(self.gcn_dim, activation='relu')(time_avg_layer)
            aq_neighbor = Dropout(self.dropoutrate)(aq_neighbor)
            return aq_neighbor




##todo: use rs as one more combined layer