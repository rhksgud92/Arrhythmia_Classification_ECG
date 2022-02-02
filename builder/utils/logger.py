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
#!/usr/bin/env python3
import os
import sys
import shutil
import copy
import logging
import logging.handlers
from collections import OrderedDict
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, roc_auc_score, average_precision_score, f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from builder.utils.metrics import Evaluator


class Logger:
    def __init__(self, args):
        self.args = args
        self.args_save = copy.deepcopy(args)
        
        # Evaluator
        self.evaluator = Evaluator(self.args)
        
        # Checkpoint and Logging Directories
        self.dir_root = os.path.join(self.args.dir_result, self.args.project_name)
        self.dir_log = os.path.join(self.dir_root, 'logs')
        self.dir_save = os.path.join(self.dir_root, 'ckpts')
        self.log_iter = args.log_iter

        if args.reset and os.path.exists(self.dir_root):
            shutil.rmtree(self.dir_root, ignore_errors=True)
        if not os.path.exists(self.dir_root):
            os.makedirs(self.dir_root)
        if not os.path.exists(self.dir_save):
            os.makedirs(self.dir_save)
        elif os.path.exists(os.path.join(self.dir_save, 'last.pth')) and os.path.exists(self.dir_log):
            shutil.rmtree(self.dir_log, ignore_errors=True)
        if not os.path.exists(self.dir_log):
            os.makedirs(self.dir_log)

        # Tensorboard Writer
        self.writer = SummaryWriter(logdir=self.dir_log, flush_secs=60)
        
        # Log variables
        self.loss = 0
        self.val_loss = 0
        self.best_auc = 0
        self.best_iter = 0
        self.best_result_so_far = np.array([])
        self.best_results = []

    # def binary_normalize(self, proba_list):
    #     return list(np.array(proba_list)/sum(proba_list))

    def log_tqdm(self, epoch, step, pbar):
        tqdm_log = "Epochs: {}, Iteration: {}, Loss: {}".format(str(epoch), str(step), str(self.loss / step))
        pbar.set_description(tqdm_log)
        
    def log_scalars(self, step):
        self.writer.add_scalar('train/loss', self.loss / step, global_step=step)
    
    def log_lr(self, lr, step):
        self.writer.add_scalar('train/lr', lr, global_step=step)

    def log_val_loss(self, val_step, step):
        self.writer.add_scalar('val/loss', self.val_loss / val_step, global_step=step)

    def add_validation_logs(self, step):
        if self.args.output_dim == 2:
            result, tpr, fnr, tnr, fpr = self.evaluator.performance_metric_binary()
            auc = result[0]
            os.system("echo  \'##### Current Validation results #####\'")
            os.system("echo  \'auc: {}, apr: {}, f1_score: {}\'".format(str(result[0]), str(result[1]), str(result[2])))
            os.system("echo  \'tpr: {}, fnr: {}, tnr: {}, fpr: {}\'".format(str(tpr), str(fnr), str(tnr), str(fpr)))

            self.writer.add_scalar('val/auc', result[0], global_step=step)
            self.writer.add_scalar('val/apr', result[1], global_step=step)
            self.writer.add_scalar('val/f1', result[2], global_step=step)
            self.writer.add_scalar('val/tpr', tpr, global_step=step)
            self.writer.add_scalar('val/fnr', fnr, global_step=step)
            self.writer.add_scalar('val/tnr', tnr, global_step=step)
            self.writer.add_scalar('val/fpr', fpr, global_step=step)

            if self.best_auc < auc:
                self.best_iter = step
                self.best_auc = auc
                self.best_result_so_far = result
                self.best_results = [tpr, fnr, tnr, fpr]

            os.system("echo  \'##### Best Validation results in history #####\'")
            os.system("echo  \'auc: {}, apr: {}, f1_score: {}\'".format(str(self.best_result_so_far[0]), str(self.best_result_so_far[1]), str(self.best_result_so_far[2])))
            os.system("echo  \'tpr: {}, fnr: {}, tnr: {}, fpr: {}\'".format(str(self.best_results[0]), str(self.best_results[1]), str(self.best_results[2]), str(self.best_results[3])))

        else:
            result, result_aucs, result_aprs, result_f1scores, tprs, fnrs, tnrs, fprs, fdrs, ppvs = self.evaluator.performance_metric_multi()

            multi_weighted_auc = result[0]
            multi_unweighted_auc = result[1]
            multi_weighted_apr = result[2]
            multi_unweighted_apr = result[3]
            multi_weighted_f1_score = result[4]
            multi_unweighted_f1_score = result[5]

            os.system("echo  \'##### Current Validation results #####\'")
            os.system("echo  \'multi_weighted: auc: {}, apr: {}, f1_score: {}\'".format(str(result[0]), str(result[2]), str(result[4])))
            os.system("echo  \'multi_unweighted: auc: {}, apr: {}, f1_score: {}\'".format(str(result[1]), str(result[3]), str(result[5])))
            os.system("echo  \'##### Each class Validation results #####\'")
            seizure_list = self.args.num_to_seizure_items
            results = []
            results.append("Label:bckg auc:{} apr:{} f1:{} tpr:{} fnr:{} tnr:{} fpr:{} fdr:{} ppv:{}".format(
                                                                                str(result_aucs[0]), str(result_aprs[0]), str(result_f1scores[0]), 
                                                                                str(tprs[0]), str(fnrs[0]), str(tnrs[0]), str(fprs[0]), str(fdrs[0]), str(ppvs[0])))
            for idx, seizure in enumerate(seizure_list):
                results.append("Label:{} auc:{} apr:{} f1:{} tpr:{} fnr:{} tnr:{} fpr:{} fdr:{} ppv:{}".format(seizure,
                                                                                str(result_aucs[idx+1]), str(result_aprs[idx+1]), str(result_f1scores[idx+1]), 
                                                                                str(tprs[idx+1]), str(fnrs[idx+1]), str(tnrs[idx+1]), str(fprs[idx+1]), 
                                                                                str(fdrs[idx+1]), str(ppvs[idx+1])))

            for i in results:
                os.system("echo  \'{}\'".format(i))

            self.writer.add_scalar('val/multi_weighted_auc', multi_weighted_auc, global_step=step)
            self.writer.add_scalar('val/multi_weighted_apr', multi_weighted_apr, global_step=step)
            self.writer.add_scalar('val/multi_weighted_f1_score', multi_weighted_f1_score, global_step=step)
            self.writer.add_scalar('val/multi_unweighted_auc', multi_unweighted_auc, global_step=step)
            self.writer.add_scalar('val/multi_unweighted_apr', multi_unweighted_apr, global_step=step)
            self.writer.add_scalar('val/multi_unweighted_f1_score', multi_unweighted_f1_score, global_step=step)

            if self.best_auc < multi_weighted_auc:
                self.best_iter = step
                self.best_auc = multi_weighted_auc
                self.best_result_so_far = result
                self.best_results = results

            os.system("echo  \'##### Best Validation results in history #####\'")
            os.system("echo  \'multi_weighted: auc: {}, apr: {}, f1_score: {}\'".format(str(self.best_result_so_far[0]), str(self.best_result_so_far[2]), str(self.best_result_so_far[4])))
            os.system("echo  \'multi_unweighted: auc: {}, apr: {}, f1_score: {}\'".format(str(self.best_result_so_far[1]), str(self.best_result_so_far[3]), str(self.best_result_so_far[5])))
            for i in self.best_results:
                os.system("echo  \'{}\'".format(i))

        self.writer.flush()

    def save(self, model, optimizer, step, epoch, last=None):
        ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_step': step, 'last_step' : last, 'score' : self.best_auc, 'epoch' : epoch}
        
        if step == self.best_iter:
            self.save_ckpt(ckpt, 'best.pth')
            
        if last:
            self.evaluator.get_attributions()
            self.save_ckpt(ckpt, 'last.pth')
    
    def save_ckpt(self, ckpt, name):
        torch.save(ckpt, os.path.join(self.dir_save, name))

    def test_result_only(self):
        result, tpr, fnr, tnr, fpr = self.evaluator.performance_metric_binary()

        # pred_time_list = self.evaluator.performance_metric_binary()

        os.system("echo  \'##### Test results #####\'")
        os.system("echo  \'auc: {}, apr: {}, f1_score: {}\'".format(str(result[0]), str(result[1]), str(result[2])))
        os.system("echo  \'tpr: {}, fnr: {}, tnr: {}, fpr: {}\'".format(str(tpr), str(fnr), str(tnr), str(fpr)))
        