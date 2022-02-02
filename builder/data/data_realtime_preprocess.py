# Copyright (c) 2021, Kwanhyung Lee. All rights reserved.
#
# Licensed under the MIT License; 
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import itertools
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import scipy.stats as ss
from tqdm import tqdm
from scipy.io.wavfile import write
from control.config import args
from itertools import groupby
import xmltodict
import json
import base64

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import torchaudio

from builder.utils.utils import *

# def collate_fn_numerics(train_data):
#     for pkl_path in train_data:
#         with open(pkl_path, 'rb') as _f:
#             data_pkl = pkl.load(_f)
#             data = data_pkl['refined_ecg_data']
#             II_signal = data[0][1,:]

        
def collate1(train_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    batch = []
    # targets = []
    
    for pkl_path in train_data:
        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)
            data = data_pkl['refined_ecg_data']
            II_signal = data[0][1,:]

            if len(II_signal) < 5000:
                len_diff = 5000 - len(II_signal)
                II_signal = np.pad(II_signal, (0,len_diff), 'constant', constant_values=0)
            elif len(II_signal) > 5000:
                II_signal = II_signal[:5000]
            else:
                pass
            batch.append((II_signal, data[1]))
            # batch.append(II_signal)
            # targets.append(data[2])

    batch_size = len(batch)
    seqs = torch.zeros(batch_size, 5000)
    targets = torch.zeros(batch_size).to(torch.long)
    for x in range(batch_size):
        sample = batch[x]
        tensor1 = sample[0]
        target1 = sample[1]
        seqs[x] = torch.tensor(tensor1)
        targets[x] = torch.tensor(target1)

    return seqs, targets

def collate2(train_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    batch = []
    # targets = []
    
    for pkl_path in train_data:
        with open(pkl_path, 'rb') as _f:
            data_pkl = pkl.load(_f)
            data = data_pkl['refined_ecg_data']

            samples = []
            for idx in range(8):
                signal = data[0][idx,:]
                if len(signal) < 5000:
                    len_diff = 5000 - len(signal)
                    signal = np.pad(signal, (0,len_diff), 'constant', constant_values=0)
                elif len(signal) > 5000:
                    signal = signal[:5000]
                else:
                    pass
                samples.append(signal)

            batch.append((np.asarray(samples), data[1]))

    batch_size = len(batch)
    seqs = torch.zeros(batch_size, 8, 5000)
    targets = torch.zeros(batch_size).to(torch.long)
    for x in range(batch_size):
        sample = batch[x]
        tensor1 = sample[0]
        target1 = sample[1]
        seqs[x] = torch.tensor(tensor1)
        targets[x] = torch.tensor(target1)

    return seqs, targets


def collate2_test(train_data):
    def seq_length_(p):
        return len(p[0])
    def target_length_(p):
        return len(p[1])

    batch = []
    
    for data in train_data:
        samples = []
        for idx in range(8):
            signal = data[0][idx,:]
            if len(signal) < 5000:
                len_diff = 5000 - len(signal)
                signal = np.pad(signal, (0,len_diff), 'constant', constant_values=0)
            elif len(signal) > 5000:
                signal = signal[:5000]
            else:
                pass
            samples.append(signal)

        batch.append((np.asarray(samples), data[1]))

    batch_size = len(batch)
    seqs = torch.zeros(batch_size, 8, 5000)
    targets = torch.zeros(batch_size).to(torch.long)
    for x in range(batch_size):
        sample = batch[x]
        tensor1 = sample[0]
        target1 = sample[1]
        seqs[x] = torch.tensor(tensor1)
        targets[x] = torch.tensor(target1)

    return seqs, targets


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, data, data_type="training dataset"):
        self._data_list = []
        self._type_list = []
        
        # qrs_slice_pad = np.pad(qrs_slice, (0,len_diff), 'constant', constant_values=0)
        for idx, pkl_path in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            target = pkl_path.split("_")[-1]
            if "1" in target:
                self._type_list.append(1)
            else:
                self._type_list.append(0)
            self._data_list.append(pkl_path)
        print("Number of positive samples: {}".format(str(self._type_list.count(1))))
        print("Number of negative samples: {}".format(str(self._type_list.count(0))))
        print("Dataset Prepared...\n")

    def __repr__(self):
        return (f"Data path: {self._data_pkl}")

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, index):
        _input = self._data_list[index]

        return _input

class Dataset_test(torch.utils.data.Dataset):

    def __init__(self, args, data, data_type="test dataset"):
        self._data_list = []
        self._type_list = []
        
        # qrs_slice_pad = np.pad(qrs_slice, (0,len_diff), 'constant', constant_values=0)
        for idx, one_data in enumerate(tqdm(data, desc="Loading files of {}...".format(data_type))):
            target = one_data[1]
            if target == 1:
                self._type_list.append(1)
            else:
                self._type_list.append(0)
            self._data_list.append((one_data[0], one_data[1]))
        print("Number of positive samples: {}".format(str(self._type_list.count(1))))
        print("Number of negative samples: {}".format(str(self._type_list.count(0))))
        print("Dataset Prepared...\n")

    def __repr__(self):
        return (f"Data path: {self._data_pkl}")

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, index):
        _input = self._data_list[index]

        return _input


def get_data_preprocess(args):
    train_data_path = args.data_path + "/train"
    validation_data_path = args.data_path + "/validation"

    train_data_list = search_walk({'path': train_data_path, 'extension': ".pkl"})
    validation_data_list = search_walk({'path': validation_data_path, 'extension': ".pkl"})
    random.shuffle(train_data_list)
    random.shuffle(validation_data_list)

    train_data = Dataset(args, data=train_data_list, data_type="training dataset")
    val_data = Dataset(args, data=validation_data_list, data_type="validation dataset")

    if args.collate == 1:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True,
                        num_workers=1, pin_memory=True, shuffle=True, collate_fn=collate1)               
        val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True,
                        num_workers=1, pin_memory=True, collate_fn=collate1)                
    elif args.collate == 2:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True,
                        num_workers=1, pin_memory=True, shuffle=True, collate_fn=collate2)               
        val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True,
                        num_workers=1, pin_memory=True, collate_fn=collate2)     
             
    return train_loader, val_loader


def get_data_preprocess_test(args):
    print("########################## ECG test dataset preprocess (pick 8 leads, zero-pad or slice to 5000 size, etc.) ##########################")
    arrhythmia_data_dirs = search_walk({'path': args.arrhythmia_test_dir, 'extension': ".xml"})
    normal_data_dirs = search_walk({'path': args.normal_test_dir, 'extension': ".xml"})
    final_data_list = []
    test_final_data_list = []
    _type_list = []
    error_count = 0
    sample = np.array([])

    for idx, pkl in enumerate(tqdm(arrhythmia_data_dirs, desc="Loading files from {}...".format("arrhythmia_data_dirs"))):
        file_unique_id = pkl.split("/")[-1].split(".")[0]
        flag = False            
        file = open(pkl,mode='r')
        xml_dict = xmltodict.parse(file.read())
        json_type = json.dumps(xml_dict)
        dict2_type = json.loads(json_type)
        dict_data = dict2_type['RestingECG']['Waveform']

        try:
            sample = [[], [], [], [], [], [], [], []]
            if len(dict_data) == 2:
                for d in dict_data:
                    if d["WaveformType"] == "Median":
                        continue
                    # sample = [(i["LeadID"], np.frombuffer(base64.b64decode(i["WaveFormData"]), dtype='int16')) for i in d["LeadData"] if i["LeadID"] in args.selected_leads]
                    lead_count = 0
                    for i in d["LeadData"]:
                        if i["LeadID"] in args.selected_leads:
                            sample[args.selected_leads.index(i["LeadID"])] = np.frombuffer(base64.b64decode(i["WaveFormData"]), dtype='int16')
                            if sample[args.selected_leads.index(i["LeadID"])].size < 1000:
                                pass
                            else:
                                lead_count += 1
                    npsample = np.asarray(sample)

            else:
                # sample = np.array([np.frombuffer(base64.b64decode(i["WaveFormData"]), dtype='int16') for i in dict_data["LeadData"] if i["LeadID"] in args.selected_leads])
                lead_count = 0
                for i in dict_data["LeadData"]:
                    if i["LeadID"] in args.selected_leads:
                        sample[args.selected_leads.index(i["LeadID"])] = np.frombuffer(base64.b64decode(i["WaveFormData"]), dtype='int16')
                        if sample[args.selected_leads.index(i["LeadID"])].size < 1000:
                            pass
                        else:
                            lead_count += 1
                npsample = np.asarray(sample)

            flag =True                            

        except:
            error_count += 1

        if flag == True:
            if lead_count == 8:
                final_data_list.append((npsample, 1, file_unique_id))
                _type_list.append(1)
    
    for idx, pkl in enumerate(tqdm(normal_data_dirs, desc="Loading files from {}...".format("normal_data_dirs"))):
        file_unique_id = pkl.split("/")[-1].split(".")[0]
        flag = False            
        file = open(pkl,mode='r')
        xml_dict = xmltodict.parse(file.read())
        json_type = json.dumps(xml_dict)
        dict2_type = json.loads(json_type)
        dict_data = dict2_type['RestingECG']['Waveform']

        try:
            sample = [[], [], [], [], [], [], [], []]
            if len(dict_data) == 2:
                for d in dict_data:
                    if d["WaveformType"] == "Median":
                        continue
                    lead_count = 0
                    for i in d["LeadData"]:
                        if i["LeadID"] in args.selected_leads:
                            sample[args.selected_leads.index(i["LeadID"])] = np.frombuffer(base64.b64decode(i["WaveFormData"]), dtype='int16')
                            if sample[args.selected_leads.index(i["LeadID"])].size < 1000:
                                pass
                            else:
                                lead_count += 1
                    npsample = np.asarray(sample)
            else:
                lead_count = 0
                for i in dict_data["LeadData"]:
                    if i["LeadID"] in args.selected_leads:
                        sample[args.selected_leads.index(i["LeadID"])] = np.frombuffer(base64.b64decode(i["WaveFormData"]), dtype='int16')
                        if sample[args.selected_leads.index(i["LeadID"])].size < 1000:
                            pass
                        else:
                            lead_count += 1
                npsample = np.asarray(sample)

            flag =True                            

        except:
            error_count += 1

        if flag == True:
            if lead_count == 8:
                final_data_list.append((npsample, 0, file_unique_id))
                _type_list.append(0)
            else:
                print("Not enough leads!!!")

    print("Error Count: ", error_count)    
    print("Number of positive samples: {}".format(str(_type_list.count(1))))
    print("Number of negative samples: {}".format(str(_type_list.count(0))))
    print("Dataset Prepared...\n")

    for data in final_data_list:
        new_data = {}
        new_data["refined_ecg_data"] = data
        if len(data[0].shape) == 1:
            pass
        else:
            if data[0].shape[1] < 4000:
                print(data[0].shape)
            else:
                test_final_data_list.append(data)

    test_data = Dataset_test(args, data=test_final_data_list, data_type="test dataset")

    test_loader = DataLoader(test_data, batch_size=args.batch_size, drop_last=True,
                        num_workers=1, pin_memory=True, collate_fn=collate2_test)      

    return test_loader