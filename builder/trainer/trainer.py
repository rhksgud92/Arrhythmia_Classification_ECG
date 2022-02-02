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
import os
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
import datetime
from control.config import args


def binary_classification(args, iteration, train_x, train_y, model, logger, device, scheduler=None, optimizer=None, criterion=None, flow_type="train"):
    iter_loss = []
    val_loss = []

    train_y = train_y.type(torch.FloatTensor)
    final_target = torch.round(train_y).type(torch.LongTensor).squeeze()

    if flow_type == "train":
        optimizer.zero_grad()

    model.init_state(device)
    logits = model(train_x)
    logits = logits.type(torch.FloatTensor)

    if flow_type == "train":
        loss = criterion(logits, final_target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        scheduler.step(iteration)    
        logger.log_lr(scheduler.get_lr()[0], iteration)
        
    else:
        # proba = nn.functional.softmax(logits, dim=1)
        loss = criterion(logits, final_target)
        loss = torch.mean(loss)
        logger.evaluator.add_batch(np.array(final_target.cpu()), np.array(logits.cpu()))

    return model, loss

