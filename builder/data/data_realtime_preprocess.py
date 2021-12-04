import datetime
import random
import itertools
import speechpy
import numpy as np
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
from tqdm import tqdm
from scipy.io.wavfile import write
from scipy.signal import stft, hilbert, butter, freqz, filtfilt, find_peaks, iirnotch
from control.config import args
from itertools import groupby
import neurokit2 as nk

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import torchaudio

from builder.utils.utils import *


def split_groups(_list, n_groups):
    k, m = divmod(len(_list), n_groups)
    return (_list[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n_groups))

def flatten(t):
    return [item for sublist in t for item in sublist]


class Dataset_hourly(torch.utils.data.Dataset):
    def __init__(self, args, pos_data, neg_data, normalization=None, data_type="training dataset"):
        print("Preparing Hicardi Dataset...")
        self._data_list = []
        self._type_list = []
        all_data = pos_data + neg_data

        for idx, data in enumerate(tqdm(all_data, desc="Loading files of {}".format(data_type))):
            for data_slice in data:
                if args.model_type == "1":
                    # static_data:  [ sex age temp spo2 ]
                    # delta_vs: [ sbp dbp hr rr gcs ] (diff)
                    # current_vs: [ sbp dbp hr rr gcs ]
                    static_data, current_vs, delta_vs, target, time_hour, pat_id = data_slice
                    vs_data = list(static_data)+list(delta_vs)+list(current_vs)
                    self._type_list.append(target)
                    self._data_list.append((torch.Tensor(vs_data), target, time_hour, int(pat_id.split("-")[-1])))

                elif args.model_type == "2":
                    # static_data:  [ sex age temp spo2 ]
                    # accumulate_data: stacked([ sbp dbp hr rr gcs ] (diff) and [ sbp dbp hr rr gcs ])
                    static_data, current_vs, delta_vs, accumulate_data, acc_len, target, time_hour, pat_id = data_slice
                    vs_data = list(static_data)+list(delta_vs)+list(current_vs)
                    self._type_list.append(target)
                    acc_data = torch.Tensor(list(accumulate_data))
                    self._data_list.append((torch.Tensor(vs_data), acc_data, acc_len, target, time_hour, int(pat_id.split("-")[-1])))

                else:
                    print("Error select correct model_type")
                    exit(1)
                    
        print("Number of positive patients: ", len(pos_data))
        print("Number of negative patients: ", len(neg_data))
        print("Number of positive slices: {}".format(str(self._type_list.count(1))))
        print("Number of negative slices: {}".format(str(self._type_list.count(0))))
        print("Dataset Prepared...\n")

    def __repr__(self):
        return (f"Data path: {self._data_pkl}")

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, index):
        _input = self._data_list[index]

        return _input

class Dataset_signal(torch.utils.data.Dataset):
    def __init__(self, args, pos_data, neg_data, normalization=None, data_type="training dataset"):
        print("Preparing Hicardi Dataset...")
        self._data_list = []
        self._type_list = []
        all_data = pos_data + neg_data

        for idx, data in enumerate(tqdm(all_data, desc="Loading files of {}".format(data_type))):
            for data_slice in data:
                if args.model_type == "1":
                    static_data, current_vs, delta_vs, stacked_ecg_resp_vs_data_pad, stacked_num, target, time_sec, pat_id = data_slice
                    vs_data = list(static_data)+list(delta_vs)+list(current_vs)
                    acc_data = torch.Tensor(list(stacked_ecg_resp_vs_data_pad))
                    self._type_list.append(target)
                    self._data_list.append((torch.Tensor(vs_data), acc_data, stacked_num, target, time_sec, int(pat_id.split("-")[-1])))

                elif args.model_type == "2":
                    # static_data:  [ sex age temp spo2 ]
                    # accumulate_data: stacked([ sbp dbp hr rr gcs ] (diff) and [ sbp dbp hr rr gcs ])
                    static_data, current_vs, delta_vs, accumulate_data, acc_len, target, time_sec, pat_id = data_slice
                    vs_data = list(static_data)+list(delta_vs)+list(current_vs)
                    self._type_list.append(target)
                    acc_data = torch.Tensor(list(accumulate_data))
                    self._data_list.append((torch.Tensor(vs_data), acc_data, acc_len, target, time_sec, int(pat_id.split("-")[-1])))

                elif args.model_type == "3":
                    # static_data, current_vs, delta_vs, new_signal_data, target, time_sec, pat_id = data_slice
                    static_data, current_vs, delta_vs, target, time_sec, pat_id = data_slice
                    vs_data = list(static_data)+list(delta_vs)+list(current_vs)
                    self._type_list.append(target)
                    self._data_list.append((torch.Tensor(vs_data), target, time_sec, int(pat_id.split("-")[-1])))

                else:
                    print("Error select correct model_type")
                    exit(1)

        print("Number of positive patients: ", len(pos_data))
        print("Number of negative patients: ", len(neg_data))
        print("Number of positive slices: {}".format(str(self._type_list.count(1))))
        print("Number of negative slices: {}".format(str(self._type_list.count(0))))
        print("Dataset Prepared...\n")

    def __repr__(self):
        return (f"Data path: {self._data_pkl}")

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, index):
        _input = self._data_list[index]

        return _input


class Dataset_shape(torch.utils.data.Dataset):
    def __init__(self, args, pos_data, neg_data, normalization=None, data_type="training dataset"):
        print("Preparing Hicardi Dataset...")
        self._data_list = []
        self._type_list = []
        all_data = pos_data + neg_data
        
        for idx, data in enumerate(tqdm(all_data, desc="Loading ecg & vs files of {}".format(data_type))):
            for data_slice in data:
                static_data, current_vs, delta_vs, ecg_resp_vs_list, target = data_slice
                self._type_list.append(target)
                vs_data = list(static_data)+list(delta_vs)+list(current_vs) + list(ecg_resp_vs_list)
                self._data_list.append((torch.Tensor(vs_data), ecg_slices, target))
        print("Number of positive patients: ", len(pos_data))
        print("Number of negative patients: ", len(neg_data))
        print("Number of positive slices: {}".format(str(self._type_list.count(1))))
        print("Number of negative slices: {}".format(str(self._type_list.count(0))))
        print("Dataset Prepared...\n")

    def __repr__(self):
        return (f"Data path: {self._data_pkl}")

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, index):
        _input = self._data_list[index]

        return _input


def get_data_preprocess(args, val_idx=1, k_cfv=4):
    ###### Vasso for window size 120 seconds ######
    ### for --prediction-after 24
    # Number of positive patients:  76
    # Number of negative patients:  394
    # Number of positive slices: 12672
    # Number of negative slices: 55764
    ### for --prediction-after 72
    # Number of positive patients:  82
    # Number of negative patients:  388
    # Number of positive slices: 13780
    # Number of negative slices: 54656
    ###############################################
    train_data_path = args.data_path
    crossfold_val_n = args.k_cross_fold_validation
    train_dir = search_walk({"path": train_data_path, "extension": ".pkl"})

    ############ data preparation part ############
    # AGE: 나이 
    # SEX: 성별 
    # IN_TIME: 환자가 응급실에 내원한 일시
    # ATTACH_TIME: 기기 붙힌 시간
    # DETACH_TIME: 기기 땐 시간
    # VS_INIT: 응급실 내원한 당시에 측정한 vital sign   [ sbp dbp hr rr temp spo2 gcs ]
    # VS_ATTACH:  기기 부착한 당시에 측정한 vital sign [ sbp dbp hr rr gcs ]
    # VS_1HR, VS_2HR, VS_3HR, VS_4HR, VS_5HR, VS_6HR: [ sbp dbp hr rr gcs ] 안쟀으면 [ nan nan nan nan nan ]
    # TS: 기기를 부착하고 처음으로 들어온 패킷의 시간입니다 시간 계산할때 여기를 기준으로 하면 됨
    # ECG: [ raw-signal, filtered-signal, r-peak info, filtered_r-peak info ] 250Hz
    # RESP: 25Hz
    # vasso_yn: 승합제 투여 여부
    # vasso_time: 승합제 투여 시간
    positive_pats_data_group_list = []
    negative_pats_data_group_list = []
    positive_pats_data_list = []
    negative_pats_data_list = []
    type_list = []
    k_cfv += 1
    vs_min_list = [float('inf')] * 7
    vs_max_list = [0] * 7
    target_sec_after = args.prediction_after * 3600 # hours to seconds
    ecg_freq = 250
    resp_freq = 25
    win_size_resp = args.window_size * resp_freq
    win_size_ecg = args.window_size * ecg_freq
    max_ecg_period = int(1.5 * ecg_freq)
    max_signal_data_len = (int((6 * 60 * 60) / float(args.window_size))) 
    max_signalslice_data_len = int((30 * 60) / float(args.window_size))
    max_vs_data_len = 7
    min_resp_size = args.resp_min_size * resp_freq
    hr_min_len = ecg_freq * 60
    rr_min_len = resp_freq * 60

    random.seed(args.seed)
    random.shuffle(train_dir)
    error_count = 0

    total_signal_resp = 0
    used_signal_resp = 0
    total_signal_ecg = 0
    used_signal_ecg = 0
    vs_hours = ['VS_1HR', 'VS_2HR', 'VS_3HR', 'VS_4HR', 'VS_5HR', 'VS_6HR']
    for idx, data_pkl in enumerate(tqdm(train_dir, desc="Preparing files")):
        pat_flag = False
        first_appending = True
        data_list = []
        accumulate_data = []
        pat_id = data_pkl.split("/")[-1].split(".")[0]
        with open(data_pkl, 'rb') as _f:
            data_pkl = pkl.load(_f)
            age = data_pkl['AGE']
            sex = data_pkl['SEX']
            IN_TIME = data_pkl['IN_TIME']
            ATTACH_TIME = data_pkl['ATTACH_TIME']
            DETACH_TIME = data_pkl['DETACH_TIME']
            VS_INIT = data_pkl['VS_INIT']
            VS_ATTACH = data_pkl['VS_ATTACH']
            TS = data_pkl['TS']
            filtered_ECG = data_pkl['ECG'][:,1]
            f_rpeaks_ECG = data_pkl['ECG'][:,3]
            RESP = data_pkl['RESP']
            VASSO_YN = data_pkl['VASSO_YN']
            VASSO_TIME = data_pkl['VASSO_TIME']

        vital_data = []
        vital_data.append(VS_ATTACH)
        for hour in vs_hours:
            v_data = data_pkl[hour]
            try:  
                _ = np.sum(v_data[0])            
                flag = True
            except:
                flag = False
            if (np.isnan(v_data).any()) or (flag == False):
                vital_data.append(vital_data[-1])
            else:
                vital_data.append(v_data)
        for i in vital_data[0][0]:
            if isinstance(i, str):
                vital_data[0] = vital_data[1]

        ### filtering vs data ###
        for idx, i in enumerate(vital_data):
            if (i[0][3] > 50) or (i[0][3] < 2):
                if idx < 5:
                    vital_data[idx][0][3] = vital_data[idx+1][0][3]
                else:
                    vital_data[idx][0][3] = vital_data[idx-1][0][3]
            if (i[0][2] > 250) or (i[0][2] < 20):
                if idx < 5:
                    vital_data[idx][0][2] = vital_data[idx+1][0][2]
                else:
                    vital_data[idx][0][2] = vital_data[idx-1][0][2]
        ##########################
        initial_vs = VS_INIT
        if isinstance(sex, str):
            if 'F' in sex:
                sex = 0.0
            else:
                sex = 1.0
        static_data = [age, sex, initial_vs[0][4], initial_vs[0][5]] # age, sex, temp, spo2  
        initial_vs = np.concatenate((initial_vs[0][:4], initial_vs[0][6]), axis=None)

        if args.data_type == "hourly":
            for time_hour, vital in enumerate(vital_data): 
                if time_hour >= 7:
                    continue

                current_vs = vital[0]
                ### filtering vs data ###
                if (initial_vs[2] > 250) or (initial_vs[2] < 20):
                    initial_vs[2] = current_vs[2]
                if (initial_vs[3] > 50) or (initial_vs[3] < 2):
                    initial_vs[3] = current_vs[3]
                ### filtering vs data ###

                if VASSO_YN == 0:
                    target = 0
                else:
                    if ((VASSO_TIME-TS).total_seconds()-(time_hour*3600)) <= target_sec_after:
                        target = 1
                        pat_flag = True
                    else:
                        target = 0
                delta_vs = initial_vs - current_vs

                if args.model_type == "1":
                    # static_data: age, sex, temp, spo2 
                    # current_vs: sbp dbp hr rr gcs
                    # delta_vs: sbp dbp hr rr gcs (처음 붙였을때와의 차이)
                    # time_hour: 시간대
                    # pat_id: 환자  id
                    data_list.append((static_data, current_vs, delta_vs, target, time_hour, pat_id))
                    type_list.append(target)
                elif args.model_type == "2":
                    # static_data: age, sex, temp, spo2 
                    # accumulate_data: current and delta (sbp dbp hr rr gcs) accumulated
                    # time_hour: 시간대
                    # pat_id: 환자  id
                    accumulate_data.append(np.concatenate((current_vs, delta_vs), axis=None))
                    stacked_vs_data = np.stack( accumulate_data, axis=0 )
                    len_diff = max_vs_data_len - stacked_vs_data.shape[0]
                    stacked_vs_data_pad = np.pad(stacked_vs_data, [(0,len_diff), (0,0)], 'constant', constant_values=0)
                    data_list.append((static_data, current_vs, delta_vs, stacked_vs_data_pad, stacked_vs_data.shape[0], target, time_hour, pat_id))
                    type_list.append(target)
                
        elif args.data_type == "signal":
            resp_size = RESP.size
            # accumulated_signal_vs_data = []
            # accumulate_data = []

            for index, resp_idx in enumerate(range(0, resp_size, win_size_resp)):
                ecg_lens = []
                resp_lens = []
                resp_indxs = []
                ecg_mins = []
                ecg_maxs = []
                time_sec = (resp_idx // resp_freq) + args.window_size
                ecg_idx = resp_idx * 10
                if (time_sec // 3600) >= 6:
                    continue

                # if resp_idx % (3600 * resp_freq) == 0:
                    # print("manually recorded: ", vital_data[time_sec // 3600][0])
                current_vs = list(vital_data[time_sec // 3600][0])
                hourly_rr = current_vs[3]
                ### filtering vs data ###
                if (initial_vs[2] > 250) or (initial_vs[2] < 20):
                    initial_vs[2] = current_vs[2]
                if (initial_vs[3] > 50) or (initial_vs[3] < 2):
                    initial_vs[3] = current_vs[3]
                ### filtering vs data ###
                delta_vs = initial_vs - current_vs
                sliced_resp = RESP[resp_idx : resp_idx + win_size_resp]
                if len(sliced_resp) < win_size_resp:
                    continue
                sliced_ecg = filtered_ECG[ecg_idx : ecg_idx + win_size_ecg]
                sliced_f_rpeaks_ECG = f_rpeaks_ECG[ecg_idx : ecg_idx + win_size_ecg]
                if np.count_nonzero(sliced_f_rpeaks_ECG) < (args.window_size//1.5): # at least once in 1.5 seconds
                    continue

                index_R = np.argwhere(sliced_f_rpeaks_ECG==1)
                if index_R[0] == 1.0:
                    pass
                else:
                    index_R_temp = np.insert(index_R, 0, 1.0)
                
                total_signal_resp += index_R_temp[-1]//10
                total_signal_ecg += index_R_temp[-1]
                
                middle_points = np.ediff1d(index_R_temp)
                RR_middle_indices = middle_points//2 + index_R_temp[:-1]

                Resp_middle_indices = RR_middle_indices//10
                median_qrs_slice_len = np.median(np.ediff1d(RR_middle_indices))
                
                for idx2, qrs_slice_index in enumerate(RR_middle_indices):
                    if (idx2 != (len(RR_middle_indices)-1)):
                        if (RR_middle_indices[idx2+1] - RR_middle_indices[idx2]) > (median_qrs_slice_len * 1.5):
                            continue
                        else:
                            qrs_slice = sliced_ecg[qrs_slice_index:RR_middle_indices[idx2+1]]
                            ecg_mins.append(np.min(qrs_slice))
                            ecg_maxs.append(np.max(qrs_slice))
                
                median_slice_min = np.median(ecg_mins)
                median_slice_max = np.median(ecg_maxs)

                resp_idx_start = Resp_middle_indices[0]
                for idx3, qrs_slice_index in enumerate(RR_middle_indices):
                    resp_idx_end = Resp_middle_indices[idx3]

                    if (idx3 != (len(RR_middle_indices)-1)):
                        if (RR_middle_indices[idx3+1] - RR_middle_indices[idx3]) > (median_qrs_slice_len * 1.5):
                            if (resp_idx_end - resp_idx_start) >= min_resp_size:
                                resp_indxs.append((resp_idx_start, resp_idx_end, resp_idx_end-resp_idx_start))
                            resp_idx_start = Resp_middle_indices[idx3+1]
                            continue
                        else:
                            qrs_slice = sliced_ecg[qrs_slice_index:RR_middle_indices[idx3+1]]
                            # below needs inspection
                            if np.absolute(median_slice_min-np.min(qrs_slice)) > (2*np.absolute(median_slice_min)):
                                if (resp_idx_end - resp_idx_start) >= min_resp_size:
                                    resp_indxs.append((resp_idx_start, resp_idx_end, resp_idx_end-resp_idx_start))
                                resp_idx_start = Resp_middle_indices[idx3+1]
                                continue

                            if np.absolute(median_slice_max-np.max(qrs_slice)) > (2*np.absolute(median_slice_max)):
                                if (resp_idx_end - resp_idx_start) >= min_resp_size:
                                    resp_indxs.append((resp_idx_start, resp_idx_end, resp_idx_end-resp_idx_start))
                                resp_idx_start = Resp_middle_indices[idx3+1]
                                continue
                                
                            ecg_len = len(qrs_slice)
                            ecg_lens.append(ecg_len)
                            # signals, info = nk.rsp_process(np.transpose(sliced_resp)[0], sampling_rate=25)
                            # nk.rsp_plot(signals, sampling_rate=25)
                            # sliced_ecg = filtered_ECG[ecg_idx : ecg_idx + win_size_ecg]
                    else:
                        if (resp_idx_end - resp_idx_start) >= min_resp_size:
                            resp_indxs.append((resp_idx_start, resp_idx_end, resp_idx_end-resp_idx_start))
                if len(ecg_lens) < (args.window_size//1.5): # at least once in 1.5 seconds
                    continue 

                if len(resp_indxs) > 0:
                    resp_beg_end = max(resp_indxs, key=lambda x: int(x[2]))
                    resp_begin_index, resp_end_index, _ = resp_beg_end
                    sliced_resp_temp = sliced_resp[resp_begin_index:resp_end_index]
                    signal_hr = round(float(hr_min_len) / np.mean(ecg_lens), 2)
                    used_signal_ecg += np.sum(ecg_lens)
                    signal_hr_std = round(np.std(ecg_lens), 2)
                    # print("1: ", signal_hr)

                    try:
                        rsp_cleaned = nk.rsp_clean(np.transpose(sliced_resp_temp)[0], sampling_rate=25)
                        # signals, info = nk.rsp_process(np.transpose(sliced_resp_temp)[0], sampling_rate=25)
                        signals, info = nk.rsp_process(rsp_cleaned, sampling_rate=25)
                        # resp_rate_median = round(np.median(signals["RSP_Rate"]), 2)
                        # print("hourly_rr: ", hourly_rr)
                        # resp_filtering
                        resp_len_temp = signals["RSP_Rate"]
                        big_threshold = hourly_rr * 1.5
                        small_threshold = hourly_rr * 0.5
                        filtered_resp_rates = [resp_one for resp_one in resp_len_temp if (resp_one <= big_threshold and resp_one >= small_threshold)]

                        total_resp_count = len(resp_len_temp)
                        used_resp_count = len(filtered_resp_rates)
                        used_signal_resp += (np.sum(len(sliced_resp_temp))) * (used_resp_count / float(total_resp_count))

                        resp_rate_median = round(np.median(filtered_resp_rates),2)
                        resp_rate_std = round(np.std(signals["RSP_Rate"]), 2)
                        
                        if np.isnan(resp_rate_median):
                            resp_rate_median = hourly_rr
                        # print("2: ", resp_rate_median)
                        # if resp_rate_std > 6:
                        #     continue

                        # rsp_rate_onsets = nk.rsp_rate(rsp_cleaned, sampling_rate=25, window=10, hop_size=5, method="trough")
                        # rsp_rate_xcorr = nk.rsp_rate(rsp_cleaned, sampling_rate=25, window=10, hop_size=5, method="xcorr")
                        # signals, info = nk.rsp_process(rsp_cleaned, sampling_rate=25)
                        # nk.rsp_plot(signals, sampling_rate=25)

                        # hr_diff = round(initial_vs[2] - signal_hr, 2)
                        # rr_diff = round(initial_vs[3] - resp_rate_median, 2)
                        # ecg_resp_vs_list.append((signal_hr, resp_rate_median, hr_diff, rr_diff))
                        # stacked_ecg_resp_vs_data = np.stack( ecg_resp_vs_list, axis=0 )
                    except:
                        pass

                    if VASSO_YN == 0:
                        target = 0
                    else:
                        if ((VASSO_TIME-TS).total_seconds()-time_sec) <= target_sec_after:
                            target = 1
                            pat_flag = True
                        else:
                            target = 0
                    
                    # static_data: age, sex, temp, spo2 
                    # current_vs: sbp dbp hr rr gcs
                    # delta_vs: sbp dbp hr rr gcs (처음 붙였을때와의 차이)
                    # ecg_resp_vs_list: (np.mean(ecg_lens), np.std(ecg_lens, resp_rate_median, resp_rate_std)
                    # time_sec: 시간대
                    # pat_id: 환자 id
                    # if args.model_type == "1":
                    #     if max_signalslice_data_len >= stacked_ecg_resp_vs_data.shape[0]:
                    #         len_diff = max_signalslice_data_len - stacked_ecg_resp_vs_data.shape[0]
                    #         stacked_ecg_resp_vs_data_pad = np.pad(stacked_ecg_resp_vs_data, [(0,len_diff), (0,0)], 'constant', constant_values=0)
                    #         stacked_num = stacked_ecg_resp_vs_data.shape[0]
                    #     else:    
                    #         stacked_ecg_resp_vs_data_pad = stacked_ecg_resp_vs_data[-max_signalslice_data_len:, :]
                    #         stacked_num = max_signalslice_data_len
                    #     print((static_data, current_vs, delta_vs, stacked_ecg_resp_vs_data_pad, stacked_num, target, time_sec, pat_id))
                    #     data_list.append((static_data, current_vs, delta_vs, stacked_ecg_resp_vs_data_pad, stacked_num, target, time_sec, pat_id))
                    #     type_list.append(target)

                    if args.model_type == "2":
                        if first_appending:
                            delta_vs = [round(vs, 2) for vs in delta_vs]
                            current_vs = [round(vs, 2) for vs in current_vs]
                            accumulate_data.append(np.concatenate((current_vs, delta_vs), axis=None))
                            stacked_vs_data = np.stack( accumulate_data, axis=0 )
                            stacked_num = stacked_vs_data.shape[0]
                            len_diff = max_signal_data_len - stacked_num
                            stacked_vs_data_pad = np.pad(stacked_vs_data, [(0,len_diff), (0,0)], 'constant', constant_values=0)
                            data_list.append((static_data, current_vs, delta_vs, stacked_vs_data_pad, stacked_num, target, time_sec, pat_id))
                            type_list.append(target)
                            first_appending = False
                            
                        if (resp_idx+win_size_resp) % (3600 * resp_freq) == 0:
                            hourly_hr = current_vs[2]
                            hourly_rr = current_vs[3]
                        else:
                            current_vs[2] = signal_hr
                            current_vs[3] = resp_rate_median
                            delta_vs = initial_vs - current_vs
                        delta_vs = [round(vs, 2) for vs in delta_vs]
                        current_vs = [round(vs, 2) for vs in current_vs]
                        accumulate_data.append(np.concatenate((current_vs, delta_vs), axis=None))
                        stacked_vs_data = np.stack( accumulate_data, axis=0 )
                        stacked_num = stacked_vs_data.shape[0]
                        len_diff = max_signal_data_len - stacked_num
                        stacked_vs_data_pad = np.pad(stacked_vs_data, [(0,len_diff), (0,0)], 'constant', constant_values=0)
                        data_list.append((static_data, current_vs, delta_vs, stacked_vs_data_pad, stacked_num, target, time_sec, pat_id))
                        type_list.append(target)
                        # print("stacked_num: ", stacked_num)
                        # print("data_list: ", data_list[-1])

                    elif args.model_type == "3":
                        if first_appending:
                            delta_vs = [round(vs, 2) for vs in delta_vs]
                            data_list.append((static_data, current_vs, delta_vs, target, 0, pat_id))
                            type_list.append(target)
                            vs_data = list(static_data)+list(delta_vs)+list(current_vs)
                            # print("age:{} sex:{} temp:{} spo2:{} del-sbp:{} del-dbp:{} del-hr:{} del-rr:{} del-gcs:{} sbp:{} dbp:{} hr:{} rr:{} gcs:{}".format(vs_data[0], vs_data[1], vs_data[2], vs_data[3], vs_data[4], vs_data[5], vs_data[6], vs_data[7], vs_data[8], vs_data[9], vs_data[10], vs_data[11], vs_data[12], vs_data[13]))
                            # print("time_sec: {}, time_hour: {}".format(str(0), str(datetime.timedelta(seconds=0))))
                            # print(" ")
                            first_appending = False
                        if (resp_idx+win_size_resp) % (3600 * resp_freq) == 0:
                            hourly_hr = current_vs[2]
                            hourly_rr = current_vs[3]
                        else:
                            current_vs[2] = signal_hr
                            current_vs[3] = resp_rate_median
                            delta_vs = initial_vs - current_vs
                        delta_vs = [round(vs, 2) for vs in delta_vs]

                        data_list.append((static_data, current_vs, delta_vs, target, time_sec, pat_id))
                        vs_data = list(static_data)+list(delta_vs)+list(current_vs)
                        # print("age:{} sex:{} temp:{} spo2:{} del-sbp:{} del-dbp:{} del-hr:{} del-rr:{} del-gcs:{} sbp:{} dbp:{} hr:{} rr:{} gcs:{}".format(vs_data[0], vs_data[1], vs_data[2], vs_data[3], vs_data[4], vs_data[5], vs_data[6], vs_data[7], vs_data[8], vs_data[9], vs_data[10], vs_data[11], vs_data[12], vs_data[13]))
                        # print("time_sec: {}, time_hour: {}".format(str(time_sec), str(datetime.timedelta(seconds=time_sec))))
                        # print(" ")
                        type_list.append(target)
                            
                    else:
                        print("Error select correct model_type")
                        exit(1)

                        
                    # except:
                    #     # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Error with respiration data !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    #     pass
                
            # data_list.append(accumulated_signal_vs_data)
            # type_list.append(target)

        else: # signal shape data type
            resp_size = RESP.size
            for idx in range(0, resp_size, win_size_resp):
                time_sec = (idx // resp_freq) + args.window_size
                ecg_idx = idx * 10

                initial_vs = VS_INIT
                if isinstance(sex, str):
                    if 'F' in sex:
                        sex = 0.0
                    else:
                        sex = 1.0
                static_data = [age, sex, initial_vs[0][4], initial_vs[0][5]] # age, sex, temp, spo2  
                initial_vs = np.concatenate((initial_vs[0][:4], initial_vs[0][6]), axis=None)
                if (time_sec // 3600) >= 7:
                    continue
                current_vs = vital_data[time_sec // 3600][0]

                ### filtering vs data ###
                if (initial_vs[2] > 250) or (initial_vs[2] < 20):
                    initial_vs[2] = current_vs[2]
                if (initial_vs[3] > 50) or (initial_vs[3] < 2):
                    initial_vs[3] = current_vs[3]
                ### filtering vs data ###

                delta_vs = initial_vs - current_vs
                sliced_resp = RESP[idx : idx + win_size_resp]
                if len(sliced_resp) < win_size_resp:
                    continue
                
                sliced_ecg = filtered_ECG[ecg_idx : ecg_idx + win_size_ecg]
                sliced_f_rpeaks_ECG = f_rpeaks_ECG[ecg_idx : ecg_idx + win_size_ecg]
                if np.count_nonzero(sliced_f_rpeaks_ECG) < (args.window_size//1.5): # at least once in 1.5 seconds
                    continue

                ecg_slices = []
                ecg_lens = []
                ecg_mins = []
                ecg_maxs = []
            
                index_R = np.argwhere(sliced_f_rpeaks_ECG==1)
                if index_R[0] == 1.0:
                    pass
                else:
                    index_R_temp = np.insert(index_R, 0, 1.0)
                middle_points = np.ediff1d(index_R_temp)
                RR_middle_indices = middle_points//2 + index_R_temp[:-1]
                median_qrs_slice_len = np.median(np.ediff1d(RR_middle_indices))
                
                for idx, qrs_slice_index in enumerate(RR_middle_indices):
                    if (idx != (len(RR_middle_indices)-1)):
                        if (RR_middle_indices[idx+1] - RR_middle_indices[idx]) > (median_qrs_slice_len * 1.5):
                            continue
                        else:
                            qrs_slice = sliced_ecg[qrs_slice_index:RR_middle_indices[idx+1]]
                            ecg_mins.append(np.min(qrs_slice))
                            ecg_maxs.append(np.max(qrs_slice))
                
                median_slice_min = np.median(ecg_mins)
                median_slice_max = np.median(ecg_maxs)

                for idx, qrs_slice_index in enumerate(RR_middle_indices):
                    if (idx != (len(RR_middle_indices)-1)):
                        if (RR_middle_indices[idx+1] - RR_middle_indices[idx]) > (median_qrs_slice_len * 1.5):
                            continue
                        else:
                            qrs_slice = sliced_ecg[qrs_slice_index:RR_middle_indices[idx+1]]
                            if (median_slice_min-np.min(qrs_slice)) > (2*np.absolute(median_slice_min)):
                                continue
                            if (median_slice_max-np.max(qrs_slice)) > (2*np.absolute(median_slice_max)):
                                continue
                            ecg_len = len(qrs_slice)
                            if len(qrs_slice) > max_ecg_period:
                                qrs_slice = qrs_slice[:max_ecg_period]
                            elif len(qrs_slice) < max_ecg_period:
                                len_diff = max_ecg_period - len(qrs_slice)
                                qrs_slice_pad = np.pad(qrs_slice, (0,len_diff), 'constant', constant_values=0)
                            ecg_slices.append(qrs_slice_pad)
                            ecg_lens.append(ecg_len)
                ecg_vs = [np.mean(ecg_lens), np.std(ecg_lens)]
                if len(ecg_slices) < (args.window_size//1.5): # at least once in 1.5 seconds
                    continue 
                            
                # ##################################
                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.subplot(14,1,1)
                # plt.plot(sliced_ecg)
                # for i in range(2, 15):
                #     plt.subplot(14,1,i)
                #     plt.plot(ecg_slices[i-1])
                # plt.show()
                # exit(1)
                # ##################################

                if VASSO_YN == 0:
                    target = 0
                else:
                    if ((VASSO_TIME-TS).total_seconds()-time_sec) <= target_sec_after:
                        target = 1
                        pat_flag = True
                    else:
                        target = 0
                # data_list.append((static_data, current_vs, delta_vs, sliced_resp, sliced_ecg, sliced_f_rpeaks_ECG, target))
                minnum_ecg_periods = int(args.window_size//1.5)
                last_ecg_slices = torch.tensor(ecg_slices[-minnum_ecg_periods:])
                data_list.append((static_data, current_vs, delta_vs, ecg_vs, last_ecg_slices, target))
                type_list.append(target)

        if pat_flag:
            positive_pats_data_list.append(data_list)
        else:
            negative_pats_data_list.append(data_list)

    positive_pats_data_group_list = list(split_groups(positive_pats_data_list, k_cfv))
    negative_pats_data_group_list = list(split_groups(negative_pats_data_list, k_cfv))
    positive_pats_traing_data = flatten(positive_pats_data_group_list[:val_idx] + positive_pats_data_group_list[val_idx+1 :-1])
    negative_pats_traing_data = flatten(negative_pats_data_group_list[:val_idx] + negative_pats_data_group_list[val_idx+1 :-1])
    print("Number of total positive patients: ", len(positive_pats_data_list))
    print("Number of total negative patients: ", len(negative_pats_data_list))
    print("All data ready...")
    
    print("1: ", total_signal_resp)
    print("2: ", total_signal_ecg)
    
    print("3: ", used_signal_resp)
    print("4: ", used_signal_ecg)
    exit(1)
    if args.data_type == "hourly":
        train_data = Dataset_hourly(args, pos_data=positive_pats_traing_data, neg_data=negative_pats_traing_data, normalization=None, data_type="training dataset")
        val_data = Dataset_hourly(args, pos_data=positive_pats_data_group_list[val_idx], neg_data=negative_pats_data_group_list[val_idx], normalization=None, data_type="validation dataset")
        test_data = Dataset_hourly(args, pos_data=positive_pats_data_group_list[-1], neg_data=negative_pats_data_group_list[-1], normalization=None, data_type="test dataset")
    elif args.data_type == "signal":
        train_data = Dataset_signal(args, pos_data=positive_pats_traing_data, neg_data=negative_pats_traing_data, normalization=None, data_type="training dataset")
        val_data = Dataset_signal(args, pos_data=positive_pats_data_group_list[val_idx], neg_data=negative_pats_data_group_list[val_idx], normalization=None, data_type="validation dataset")
        test_data = Dataset_signal(args, pos_data=positive_pats_data_group_list[-1], neg_data=negative_pats_data_group_list[-1], normalization=None, data_type="test dataset")

    # class_sample_count = np.unique(train_data._type_list, return_counts=True)[1]
    # weight = 1. / class_sample_count
    # samples_weight = weight[train_data._type_list]
    # samples_weight = torch.from_numpy(samples_weight)
    # samples_weigth = samples_weight.double()
    # sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    # train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True,
    #                 num_workers=1, pin_memory=True, sampler=sampler)               
    train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True,
                    num_workers=1, pin_memory=True, shuffle=True)               
    val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True,
                    num_workers=1, pin_memory=True)               
    test_loader = DataLoader(test_data, batch_size=args.batch_size, drop_last=True,
                    num_workers=1, pin_memory=True)  


    return train_loader, val_loader, test_loader, len(train_data._data_list), len(val_data._data_list), len(test_data._data_list)
