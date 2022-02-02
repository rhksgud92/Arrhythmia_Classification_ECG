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
import torch.quantization

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
train_loader, val_loader = get_data_preprocess(args)
model = get_model(args) # "get_model" must be after "get_data_preprocess"

# model_quantize_float16 = torch.quantization.quantize_dynamic(
#     model,
#     dtype=torch.float16
# )
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

optimizer = optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)

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
        train_x, train_y = train_batch
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        iteration += 1
        model, iter_loss = get_trainer(args, iteration, train_x, train_y, model, logger, device, scheduler, optimizer, criterion)
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

            if args.quantization:
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                # model_fused = torch.quantization.fuse_modules(model, [['conv', 'relu']])
                model_fused = model
                model_prepared = torch.quantization.prepare(model_fp32_fused)                
                input_quant = torch.randn(args.batch_size, 1, 8, 5000)
                model_prepared(input_quant)
                model_quant = torch.quantization.convert(model_prepared)
                # res = model_int8(input_fp32)
            else:
                pass


            with torch.no_grad():
                for idx, batch in enumerate(tqdm(val_loader)):
                    val_x, val_y = batch
                    val_x = val_x.to(device)
                    val_y = val_y.to(device)

                    if args.quantization:
                        model, val_loss = get_trainer(args, iteration, val_x, val_y, model_quant, logger, device, scheduler, optimizer, criterion, flow_type="test")
                    else:
                        model, val_loss = get_trainer(args, iteration, val_x, val_y, model, logger, device, scheduler, optimizer, criterion, flow_type="test")
                    logger.val_loss += val_loss
                    val_iteration += 1
                
                logger.log_val_loss(val_iteration, iteration)
                logger.add_validation_logs(iteration)

                if args.quantization:
                    logger.save(model_quant, optimizer, iteration, epoch)
                else:
                    logger.save(model, optimizer, iteration, epoch)

                # if args.quantization:
                #     logger.save(model_quantize_qint8, optimizer, iteration, epoch)
                # else:
                #     logger.save(model, optimizer, iteration, epoch)

            model.train()

    pbar.update(1)

if args.quantization:
    torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.float16
    )
else:
    pass