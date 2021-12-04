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


def binary_classification(args, iteration, train_x, train_y, model, logger, device, scheduler=None, optimizer=None, criterion=None, pat_info=None, flow_type="train"):
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
        # logger.evaluator.add_pat_info(pat_info)

    if args.patient_time:
        if args.data_type == "hourly":
            max_int = 10 * 3600
            time_unit = 3600
        else:
            max_int = 10 * 3600
            time_unit = 1

        probability = logits[:, 1]
        prediction = (probability > args.threshold).int()
        pred_time, patient_id = pat_info

        for idx, positive in enumerate(final_target):
            pat_id_str = str(patient_id[idx].numpy())
            if (positive == 1) and (prediction[idx] == 1):
                if pat_id_str not in logger.evaluator.pat_rank:
                    logger.evaluator.pat_rank[pat_id_str] = max_int
                time_sec = int(pred_time[idx] * time_unit)
                if time_sec < logger.evaluator.pat_rank[pat_id_str]:
                    logger.evaluator.pat_rank[pat_id_str] = time_sec
            
            elif positive == 1:
                if pat_id_str not in logger.evaluator.pat_rank:
                    logger.evaluator.pat_rank[pat_id_str] = max_int

    return model, loss

