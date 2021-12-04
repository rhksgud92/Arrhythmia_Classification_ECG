import numpy as np
import os
import argparse
import random
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from itertools import groupby
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torchsummary import summary

from builder.utils.lars import LARC
from control.config import args
from builder.data.data_realtime_preprocess import get_data_preprocess
from builder.models import get_model
from builder.utils.logger import Logger
from builder.utils.utils import set_seeds, set_devices
from builder.utils.cosine_annealing_with_warmup import CosineAnnealingWarmUpRestarts
from builder.utils.cosine_annealing_with_warmupSingle import CosineAnnealingWarmUpSingle
from builder.trainer import get_trainer
from builder.trainer import binary_classification

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

set_seeds(args)
device = set_devices(args)
logger = Logger(args)
# Load Data, Create Model 
train_loader, val_loader, test_loader, len_train_dir, len_val_dir, len_test_dir = get_data_preprocess(args)
model = get_model(args) # "get_model" must be after "get_data_preprocess"
model = model(args, device).to(device)
criterion = nn.CrossEntropyLoss(reduction='mean')

if args.checkpoint:
    if args.last:
        ckpt_path = args.dir_result + '/' + args.project_name + '/ckpts/best.pth'
    elif args.best:
        ckpt_path = args.dir_result + '/' + args.project_name + '/ckpts/best.pth'
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    logger.best_auc = checkpoint['score']
    start_epoch = checkpoint['epoch']
    del checkpoint
else:
    logger.best_auc = 0
    start_epoch = 1

if args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
elif args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optim == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr = args.lr_init)
elif args.optim == 'adam_lars':
    optimizer = optim.Adam(model.parameters(), lr = args.lr_init, weight_decay=args.weight_decay)
    optimizer = LARC(optimizer=optimizer, eps=1e-8, trust_coefficient=0.001)
elif args.optim == 'sgd_lars':
    optimizer = optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = LARC(optimizer=optimizer, eps=1e-8, trust_coefficient=0.001)
elif args.optim == 'adamw_lars':
    optimizer = optim.AdamW(model.parameters(), lr = args.lr_init, weight_decay=args.weight_decay)
    optimizer = LARC(optimizer=optimizer, eps=1e-8, trust_coefficient=0.001)

one_epoch_iter_num = len(train_loader)
print("Iterations per epoch: ", one_epoch_iter_num)
iteration_num = args.epochs * one_epoch_iter_num

if args.lr_scheduler == "CosineAnnealing":
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.t_0*one_epoch_iter_num, T_mult=args.t_mult, eta_max=args.lr_max, T_up=args.t_up*one_epoch_iter_num, gamma=args.gamma)
elif args.lr_scheduler == "Single":
    scheduler = CosineAnnealingWarmUpSingle(optimizer, max_lr=args.lr_init * math.sqrt(args.batch_size), epochs=args.epochs, steps_per_epoch=one_epoch_iter_num, div_factor=math.sqrt(args.batch_size))

model.train()
iteration = 0
logger.loss = 0
pbar = tqdm(total=args.epochs, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
for epoch in range(start_epoch, args.epochs+1):
    epoch_losses =[]
    loss = 0

    for train_batch in train_loader:
        if (args.data_type == "hourly") and (args.model_type == "1"):
            vs_data, train_y, time_sec_or_hour, pat_id = train_batch
            vs_data_x = vs_data.to(device)
            train_y = train_y.to(device)
            train_x = (vs_data_x)
        elif (args.data_type == "hourly") and (args.model_type == "2"):
            vs_data, accumulated_data, acc_len, train_y, time_sec_or_hour, pat_id = train_batch
            vs_data_x = vs_data.to(device)
            accumulated_data_x, acc_len_x = accumulated_data.to(device), acc_len.to(device)
            train_y = train_y.to(device)
            train_x = (vs_data_x, accumulated_data_x, acc_len_x)
        elif (args.data_type == "signal") and (args.model_type != "3"):
            vs_data, accumulated_data, acc_len, train_y, time_sec_or_hour, pat_id = train_batch
            vs_data_x = vs_data.to(device)
            accumulated_data_x, acc_len_x = accumulated_data.to(device), acc_len.to(device)
            train_y = train_y.to(device)
            train_x = (vs_data_x, accumulated_data_x, acc_len_x)
        else:
            vs_data, train_y, time_sec_or_hour, pat_id = train_batch
            train_x = vs_data.to(device)
            train_y = train_y.to(device)

        iteration += 1
        pat_info = (time_sec_or_hour, pat_id)
        model, iter_loss = get_trainer(args, iteration, train_x, train_y, model, logger, device, scheduler, optimizer, criterion, pat_info=pat_info)
        logger.loss += iter_loss

        ### LOGGING
        if iteration % args.log_iter == 0:
            logger.log_tqdm(epoch, iteration, pbar)
            logger.log_scalars(iteration)

        ### VALIDATION
        if iteration % (one_epoch_iter_num//2) == 0:
            model.eval()
            logger.evaluator.reset()
            val_iteration = 0
            logger.val_loss = 0
            with torch.no_grad():
                for idx, batch in enumerate(tqdm(val_loader)):
                    if (args.data_type == "hourly") and (args.model_type == "1"):
                        vs_data, val_y, time_sec_or_hour, pat_id = batch
                        vs_data_x, val_y = vs_data.to(device), val_y.to(device)
                        val_x = (vs_data_x)
                    elif (args.data_type == "hourly") and (args.model_type == "2"):
                        vs_data, accumulated_data, acc_len, val_y, time_sec_or_hour, pat_id = batch
                        vs_data_x, accumulated_data_x, acc_len_x, val_y = vs_data.to(device), accumulated_data.to(device), acc_len.to(device), val_y.to(device)
                        val_x = (vs_data_x, accumulated_data_x, acc_len_x)
                    elif (args.data_type == "signal") and (args.model_type != "3"):
                        vs_data, accumulated_data, acc_len, val_y, time_sec_or_hour, pat_id = batch
                        vs_data_x, accumulated_data_x, acc_len_x, val_y = vs_data.to(device), accumulated_data.to(device), acc_len.to(device), val_y.to(device)
                        val_x = (vs_data_x, accumulated_data_x, acc_len_x)
                    else:
                        vs_data, val_y, time_sec_or_hour, pat_id = batch
                        val_x = vs_data.to(device)
                        val_y = val_y.to(device)

                    pat_info = (time_sec_or_hour, pat_id)
                    model, val_loss = get_trainer(args, iteration, val_x, val_y, model, logger, device, scheduler, optimizer, criterion, pat_info=pat_info, flow_type="test")
                    logger.val_loss += val_loss
                    val_iteration += 1
                
                logger.log_val_loss(val_iteration, iteration)
                logger.add_validation_logs(iteration)
                logger.save(model, optimizer, iteration, epoch)
            model.train()

            # if logger.save_flag:
    pbar.update(1)

# # logger.save(model, optimizer, iteration)

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
logger.writer.close()
