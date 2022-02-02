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
from builder.utils.utils import search_walk
import xmltodict
import json
from tqdm import tqdm
import argparse
import base64
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def main(args):
    arrhythmia_data_dirs = search_walk({'path': args.arrhythmia_data_directory, 'extension': ".xml"})
    normal_data_dirs = search_walk({'path': args.normal_data_directory, 'extension': ".xml"})
    final_data_list = []
    _type_list = []
    error_count = 0
    sample = np.array([])
    save_folder_path = args.save_directory + "/" + args.data_type
    if os.path.isdir(args.save_directory):
        os.system("rm -rf {}".format(args.save_directory))
    os.system("mkdir {}".format(args.save_directory))
    if os.path.isdir(save_folder_path):
        os.system("rm -rf {}".format(save_folder_path))
    os.system("mkdir {}".format(save_folder_path))

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
                with open(save_folder_path + "/{}_{}.pkl".format(data[2], str(data[1])), 'wb') as _f:
                    pickle.dump(new_data, _f)      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=1004,
                        help='Random seed number')
    parser.add_argument('--selected-leads', type=list, default=["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"])
    parser.add_argument('--data-type', type=str, default="validation", choices=["train", "validation", "test"])
    #################### for --arrhythmia-data-directory directories ####################
    # train_arrhythmia_data_path = "/nfs/banner/ext01/shared/ecg/data/train/arrhythmia"
    # validation_arrhythmia_data_path = "/nfs/banner/ext01/shared/ecg/data/validation/arrhythmia"
    # parser.add_argument('--arrhythmia-data-directory', '-add', type=str,
    #                     default="/nfs/banner/ext01/shared/ecg/data/train/arrhythmia",
    #                     help='Path to ecg arrhythmia data')
    parser.add_argument('--arrhythmia-data-directory', '-add', type=str,
                        default="/nfs/banner/ext01/shared/ecg/data/validation/arrhythmia",
                        help='Path to ecg arrhythmia data')
    #################### for --normal-data-directory directories ####################
    # train_normal_data_path = "/nfs/banner/ext01/shared/ecg/data/train/normal"
    # validation_normal_data_path = "/nfs/banner/ext01/shared/ecg/data/validation/normal"
    # parser.add_argument('--normal-data-directory', '-ndd', type=str,
    #                     default="/nfs/banner/ext01/shared/ecg/data/train/normal",
    #                     help='Path to ecg normal data')
    parser.add_argument('--normal-data-directory', '-ndd', type=str,
                        default="/nfs/banner/ext01/shared/ecg/data/validation/normal",
                        help='Path to ecg normal data')                        
    # #################### for --save_directory directories ####################
    # # train_normal_label_path = "/nfs/banner/ext01/shared/ecg/label/train/normal"
    # # train_arrhythmia_label_path = "/nfs/banner/ext01/shared/ecg/label/train/arrhythmia"
    # # validation_normal_label_path = "/nfs/banner/ext01/shared/ecg/label/validation/normal"
    # # validation_arrhythmia_label_path = "/nfs/banner/ext01/shared/ecg/label/validation/arrhythmia"
    # parser.add_argument('--label-directory', '-ld', type=str,
    #                     default='/nfs/banner/ext01/shared/ecg/label/train/arrhythmia',
    #                     help='Path to label data')
    parser.add_argument('--save-directory', '-sd', type=str,
                        default='/nfs/banner/ext01/shared/ecg/converted_data_final',
                        help='Path to save converted data')

    args = parser.parse_args()
    main(args)

