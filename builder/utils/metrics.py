import os
# import shap
# import lime
import random
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, average_precision_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import torch.nn as nn
import torch


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.n_labels = args.output_dim
        self.confusion_matrix = np.zeros((self.n_labels, self.n_labels))
        self.batch_size = args.batch_size
        self.best_auc = 0
        self.labels_list = [i for i in range(self.n_labels)]

        self.pat_rank = {}
        self.pat_info = []

        self.y_true_multi = []
        self.y_pred_multi = []
        self.tnr90threshold = 0
    
    def add_pat_info(self, pat_info):
        self.pat_info.append(pat_info)

    def binary_normalize(self, i):
        proba_list = [i[0], max(i[1:])]
        return np.array(proba_list)/sum(proba_list)

    def add_batch(self, y_true, y_pred_multi):
        y_pred_final = np.argmax(y_pred_multi, axis=1)
        y_true_multi = np.zeros((len(y_true), self.args.output_dim))
        y_true_multi[range(len(y_true)), y_true] = 1
    
        self.y_pred_multi.append(y_pred_multi)
        self.y_true_multi.append(y_true_multi)

        self.confusion_matrix += confusion_matrix(y_true, y_pred_final, labels=self.labels_list)
    
    def performance_metric_binary(self):

        # print(self.pat_info)
        # for_pat_check_true = list(self.y_true_multi)
        # for_pat_check_pred = list(self.y_pred_multi)

        self.y_true_multi = np.concatenate(self.y_true_multi, 0)
        self.y_pred_multi = np.concatenate(self.y_pred_multi, 0)

        # print("self.y_true_multi: ", self.y_true_multi)
        # print("self.y_pred_multi: ", self.y_pred_multi)
        self.y_pred_multi = np.nan_to_num(self.y_pred_multi)

        auc = roc_auc_score (self.y_true_multi, self.y_pred_multi)
        apr = average_precision_score (self.y_true_multi, self.y_pred_multi)
        
        y_true_multi_array = np.argmax(self.y_true_multi, axis=1)
        f1 = 0
        for i in range(1, 200):
            threshold = 1. / i
            temp_output = np.array(self.y_pred_multi[:,1])
            temp_output[temp_output>=threshold] = 1
            temp_output[temp_output<threshold] = 0
            temp_score = f1_score(y_true_multi_array, temp_output, average="binary")
            if temp_score > f1:
                f1 = temp_score
            
        result = np.round(np.array([auc, apr, f1]), decimals=4)
        fpr, tpr, thresholds = roc_curve(y_true_multi_array, self.y_pred_multi[:,1], pos_label=1)
        fnr = 1 - tpr 
        tnr = 1 - fpr
        best_threshold = np.argmax(tpr + tnr)

        ################ patient wise prediction time check ################
        tnr90 = list(tnr)
        tnr90 = [0 if x< 0.999 else x for x in tnr90]
        tnr90_threshold = np.argmax(tpr + tnr90)

        print("tnr90::tpr: ", np.round(tpr[tnr90_threshold], decimals=4))
        print("tnr90::tnr: ", np.round(tnr[tnr90_threshold], decimals=4))
        print("tnr90::threshold: ", thresholds[tnr90_threshold])
        self.tnr90threshold = thresholds[tnr90_threshold]
        
        ################ patient wise prediction time check ################

        return result, np.round(tpr[best_threshold], decimals=4), np.round(fnr[best_threshold], decimals=4), np.round(tnr[best_threshold], decimals=4), np.round(fpr[best_threshold], decimals=4)

    def performance_metric_multi(self):
        print("##### confusion_matrix #####")
        print("bckg and {}".format(" ".join(self.args.diseases_to_train)))
        print("Left: true, Top: pred")
        row_sums = self.confusion_matrix.sum(axis=1)
        confusion_matrix_proba = self.confusion_matrix / row_sums[:, np.newaxis]
        print(confusion_matrix_proba)

        self.y_true_multi = np.concatenate(self.y_true_multi, 0)
        self.y_pred_multi = np.concatenate(self.y_pred_multi, 0)

        multi_weighted_auc = roc_auc_score (self.y_true_multi, self.y_pred_multi, average="weighted")
        multi_unweighted_auc = roc_auc_score (self.y_true_multi, self.y_pred_multi, average="macro")
        multi_aucs = roc_auc_score (self.y_true_multi, self.y_pred_multi, average=None, multi_class='ovr')

        multi_weighted_apr = average_precision_score (self.y_true_multi, self.y_pred_multi, average="weighted")
        multi_unweighted_apr = average_precision_score (self.y_true_multi, self.y_pred_multi, average="macro")
        multi_aprs = average_precision_score (self.y_true_multi, self.y_pred_multi, average=None)

        y_true_multi_array = np.argmax(self.y_true_multi, axis=1)
        y_pred_multi_array = np.argmax(self.y_pred_multi, axis=1)
        multi_weighted_f1_score = f1_score(y_true_multi_array, y_pred_multi_array, average="weighted")
        multi_unweighted_f1_score = f1_score(y_true_multi_array, y_pred_multi_array, average="macro")
        multi_f1_scores = f1_score(y_true_multi_array, y_pred_multi_array, average=None)
            
        result = np.round(np.array([multi_weighted_auc, multi_unweighted_auc, multi_weighted_apr, multi_unweighted_apr, multi_weighted_f1_score, multi_unweighted_f1_score]), decimals=4)
        result_aucs = np.round(multi_aucs, decimals=4)
        result_aprs = np.round(multi_aprs, decimals=4)
        result_f1scores = np.round(multi_f1_scores, decimals=4)

        tprs = []
        fnrs = []
        tnrs = []
        fprs = []
        fdrs = []
        ppvs = []
        row_sums = self.confusion_matrix.sum(axis=1)
        column_sums = self.confusion_matrix.sum(axis=0)
        
        for i in range(self.args.output_dim):
            tp = float(self.confusion_matrix[i][i])
            fn = float(row_sums[i] - tp)
            fp = float(column_sums[i] - tp)
            tn = float(np.sum(self.confusion_matrix) - row_sums[i] - column_sums[i] + tp)

            if (tp + fn) == 0:
                tpr = 0
                fnr = 1
            else:
                tpr = tp / (tp + fn) #sensitivity (recall)
                fnr = fn / (tp + fn)

            if (tn + fp) == 0:
                tnr = 0
                fpr = 1
            else:
                tnr = tn / (tn + fp) #specificity
                fpr = fp / (fp + tn)
            
            if (tp + fp) == 0:
                fdr = 1
                ppv = 0
            else:
                fdr = fp / (tp + fp)
                ppv = tp / (tp + fp) 

            tprs.append( np.round(tpr, decimals=4))
            fnrs.append( np.round(fnr, decimals=4))
            tnrs.append( np.round(tnr, decimals=4))
            fprs.append( np.round(fpr, decimals=4))
            fdrs.append( np.round(fdr, decimals=4))
            ppvs.append( np.round(ppv, decimals=4))
            
        return result, result_aucs, result_aprs, result_f1scores, tprs, fnrs, tnrs, fprs, fdrs, ppvs
       
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_labels, self.n_labels))
        self.y_true_multi = []
        self.y_pred_multi = []
        self.pat_info = []
