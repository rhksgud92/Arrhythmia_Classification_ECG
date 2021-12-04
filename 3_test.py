import os
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import random
import datetime

import torch
from torch.autograd import Variable
import torch.nn as nn

from control.config import args
from builder.data.data_realtime_preprocess import get_data_preprocess
from builder.models import get_model
from builder.utils.metrics import Evaluator
from builder.utils.logger import Logger
from builder.trainer.trainer import *
from builder.utils.utils import set_seeds, set_devices

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

label_method_max = True
scheduler = None
optimizer = None
criterion = nn.CrossEntropyLoss(reduction='none')
iteration = 1
set_seeds(args)
device = set_devices(args)
logger = Logger(args)
logger.loss = 0
print("Project name is: ", args.project_name)

# seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)

# Get Dataloader, Model
train_loader, val_loader, test_loader, len_train_dir, len_val_dir, len_test_dir = get_data_preprocess(args)
model = get_model(args)
model = model(args, device).to(device)
# evaluator = Evaluator(args)
names = [args.project_name]

for name in names: 
    # Check if checkpoint exists
    if args.last:
        ckpt_path = args.dir_result + '/' + name + '/ckpts/best.pth'
    elif args.best:
        ckpt_path = args.dir_result + '/' + name + '/ckpts/best.pth'
    if not os.path.exists(ckpt_path):
        continue

    ckpt = torch.load(ckpt_path, map_location=device)

    # state = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
    state = {k: v for k, v in ckpt['model'].items()}
    # print("state: ", state)
    model.load_state_dict(state)

    model.eval()
    print('loaded model')
    
    logger.evaluator.reset()
    result_list = []
    with torch.no_grad():
        for test_batch in tqdm(test_loader, total=len(test_loader), bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            if (args.data_type == "hourly") and (args.model_type == "1"):
                vs_data, test_y, time_sec_or_hour, pat_id = test_batch
                vs_data_x = vs_data.to(device)
                test_y = test_y.to(device)
                test_x = (vs_data_x)
            elif (args.data_type == "hourly") and (args.model_type == "2"):
                vs_data, accumulated_data, acc_len, test_y, time_sec_or_hour, pat_id = test_batch
                vs_data_x = vs_data.to(device)
                accumulated_data_x, acc_len_x = accumulated_data.to(device), acc_len.to(device)
                test_y = test_y.to(device)
                test_x = (vs_data_x, accumulated_data_x, acc_len_x)
            elif (args.data_type == "signal") and (args.model_type != "3"):
                vs_data, accumulated_data, acc_len, test_y, time_sec_or_hour, pat_id = test_batch
                vs_data_x = vs_data.to(device)
                accumulated_data_x, acc_len_x = accumulated_data.to(device), acc_len.to(device)
                test_y = test_y.to(device)
                test_x = (vs_data_x, accumulated_data_x, acc_len_x)
            else:
                vs_data, test_y, time_sec_or_hour, pat_id = test_batch
                test_x = vs_data.to(device)
                test_y = test_y.to(device)

            pat_info = (time_sec_or_hour, pat_id)

            ### Model Structures
            if args.trainer == "binary_classification": 
                model, _ = binary_classification(args, iteration, test_x, test_y, model, logger, device, scheduler,
                                            optimizer, criterion, pat_info=pat_info, flow_type="test")    
            else:
                print("Selected trainer is not prepared yet...")
                exit(1)

    logger.test_result_only()
    pred_result_list = list(sorted(logger.evaluator.pat_rank.items()))
    pred_time_list = []
    pat_list = []
    for pat_id, time_pred in pred_result_list:
        if time_pred == 36000:
            time = "False"
        else:
            time = str(datetime.timedelta(seconds=time_pred))
        print("{}: {}".format(str(pat_id), time))
        pred_time_list.append(time)
        pat_list.append(str(pat_id))
    for i in pat_list:
        print(i)
    for i in pred_time_list:
        print(i)


          