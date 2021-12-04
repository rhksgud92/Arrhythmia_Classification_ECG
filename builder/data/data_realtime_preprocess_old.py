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


class Dataset(torch.utils.data.Dataset):
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
                self._data_list.append((torch.Tensor(vs_data), target))
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
    min_resp_size = args.resp_min_size * resp_freq
    

    random.seed(args.seed)
    random.shuffle(train_dir)
    error_count = 0
    vs_hours = ['VS_1HR', 'VS_2HR', 'VS_3HR', 'VS_4HR', 'VS_5HR', 'VS_6HR']
    for idx, data_pkl in enumerate(tqdm(train_dir, desc="Preparing files")):
        pat_flag = False
        data_list = []
        accumulate_data = []
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

        if args.data_type == "hourly":
            for time_hour, vital in enumerate(vital_data): 
                if time_hour >= 7:
                    continue
                initial_vs = VS_INIT
                if isinstance(sex, str):
                    if 'F' in sex:
                        sex = 0.0
                    else:
                        sex = 1.0
                static_data = [age, sex, initial_vs[0][4], initial_vs[0][5]] # age, sex, temp, spo2  
                initial_vs = np.concatenate((initial_vs[0][:4], initial_vs[0][6]), axis=None)
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
                    data_list.append((static_data, current_vs, delta_vs, target))
                    type_list.append(target)
                elif args.model_type == "2":
                    accumulate_data.append(current_vs)
                    data_list.append((static_data, np.stack( accumulate_data, axis=0 ), delta_vs, target))
                    type_list.append(target)
                
        elif args.data_type == "signal":
            resp_size = RESP.size
            ecg_resp_vs_list = []
            for index, idx in enumerate(tqdm(range(win_size_resp, resp_size, win_size_resp))):
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
                # signals, info = nk.rsp_process(np.transpose(sliced_resp)[0], sampling_rate=25)
                # nk.rsp_plot(signals, sampling_rate=25)sliced_ecg = filtered_ECG[ecg_idx : ecg_idx + win_size_ecg]
                
                sliced_ecg = filtered_ECG[ecg_idx : ecg_idx + win_size_ecg]
                sliced_f_rpeaks_ECG = f_rpeaks_ECG[ecg_idx : ecg_idx + win_size_ecg]
                if np.count_nonzero(sliced_f_rpeaks_ECG) < (args.window_size//1.5): # at least once in 1.5 seconds
                    continue

                ecg_lens = []
                resp_lens = []
                resp_indxs = []
                ecg_mins = []
                ecg_maxs = []
            
                index_R = np.argwhere(sliced_f_rpeaks_ECG==1)
                if index_R[0] == 1.0:
                    pass
                else:
                    index_R_temp = np.insert(index_R, 0, 1.0)
                
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

                resp_idx_start = Resp_middle_indices[index]
                for idx3, qrs_slice_index in enumerate(RR_middle_indices):
                    resp_idx_end = Resp_middle_indices[idx3]

                    if (idx3 != (len(RR_middle_indices)-1)):
                        if (RR_middle_indices[idx3+1] - RR_middle_indices[idx3]) > (median_qrs_slice_len * 1.5):
                            if (resp_idx_end - resp_idx_start) < min_resp_size:
                                resp_idx_start = Resp_middle_indices[idx3+1]
                            else:
                                resp_indxs.append((resp_idx_start, resp_idx_end))
                                resp_idx_start = Resp_middle_indices[idx3+1]
                            continue
                        else:
                            qrs_slice = sliced_ecg[qrs_slice_index:RR_middle_indices[idx3+1]]
                            if (median_slice_min-np.min(qrs_slice)) > (2*np.absolute(median_slice_min)):
                                if (resp_idx_end - resp_idx_start) < min_resp_size:
                                    resp_idx_start = Resp_middle_indices[idx3+1]
                                else:
                                    resp_indxs.append((resp_idx_start, resp_idx_end))
                                    resp_idx_start = Resp_middle_indices[idx3+1]
                                continue

                            if (median_slice_max-np.max(qrs_slice)) > (2*np.absolute(median_slice_max)):
                                if (resp_idx_end - resp_idx_start) < min_resp_size:
                                    resp_idx_start = Resp_middle_indices[idx3+1]
                                else:
                                    resp_indxs.append((resp_idx_start, resp_idx_end))
                                    resp_idx_start = Resp_middle_indices[idx3+1]
                                continue
                                
                            ecg_len = len(qrs_slice)
                            ecg_lens.append(ecg_len)
                            # signals, info = nk.rsp_process(np.transpose(sliced_resp)[0], sampling_rate=25)
                            # nk.rsp_plot(signals, sampling_rate=25)
                            # sliced_ecg = filtered_ECG[ecg_idx : ecg_idx + win_size_ecg]
                    else:
                        resp_indxs.append((resp_idx_start, resp_idx_end))
                if len(ecg_lens) < (args.window_size//1.5): # at least once in 1.5 seconds
                    continue 

                resp_lens_trough = []
                resp_lens_xcorr = []
                for resp_beg_end in resp_indxs:
                    resp_begin_index, resp_end_index = resp_beg_end
                    sliced_resp_temp = sliced_resp[resp_begin_index:resp_end_index]

                    rsp_cleaned = nk.rsp_clean(np.transpose(sliced_resp_temp)[0], sampling_rate=25)
                    # rsp_rate_onsets = nk.rsp_rate(rsp_cleaned, sampling_rate=25, method="trough")
                    # rsp_rate_xcorr = nk.rsp_rate(rsp_cleaned, sampling_rate=25, method="xcorr")
                    rsp_rate_onsets =
                    rsp_rate_xcorr = 
                    if len(rsp_rate_onsets) > 0:
                        resp_lens_trough.append(rsp_rate_onsets)
                        resp_lens_xcorr.append(rsp_rate_xcorr)
                    # signals, info = nk.rsp_process(rsp_cleaned, sampling_rate=25)
                    # print("resp len: ", resp_end_index-resp_begin_index)
                    # print("1 rsp_rate_onsets: ", rsp_rate_onsets)
                    # print("2 rsp_rate_xcorr: ", rsp_rate_xcorr)
                    # nk.rsp_plot(signals, sampling_rate=25)
                    
                ecg_resp_vs_list.append((np.mean(ecg_lens), np.std(ecg_lens), np.mean(resp_lens_trough), np.mean(resp_lens_trough), np.mean(resp_lens_xcorr), np.mean(resp_lens_xcorr)))
                            
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
                data_list.append((static_data, current_vs, delta_vs, ecg_resp_vs_list, target))
                type_list.append(target)

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
    print("Number of total positive slices: {}".format(str(type_list.count(1))))
    print("Number of total negative slices: {}".format(str(type_list.count(0))))
    print("All data ready...")

    train_data = Dataset(args, pos_data=positive_pats_traing_data, neg_data=negative_pats_traing_data, normalization=None, data_type="training dataset")
    val_data = Dataset(args, pos_data=positive_pats_data_group_list[val_idx], neg_data=negative_pats_data_group_list[val_idx], normalization=None, data_type="validation dataset")
    test_data = Dataset(args, pos_data=positive_pats_data_group_list[-1], neg_data=negative_pats_data_group_list[-1], normalization=None, data_type="test dataset")

    class_sample_count = np.unique(train_data._type_list, return_counts=True)[1]
    weight = 1. / class_sample_count
    samples_weight = weight[train_data._type_list]
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True,
                    num_workers=1, pin_memory=True, sampler=sampler)               
    val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True,
                    num_workers=1, pin_memory=True)               
    test_loader = DataLoader(test_data, batch_size=args.batch_size, drop_last=True,
                    num_workers=1, pin_memory=True)  


    return train_loader, val_loader, test_loader, len(train_data._data_list), len(val_data._data_list), len(test_data._data_list)
