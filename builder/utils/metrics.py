import os
import random
import numpy as np
import pickle as pkl
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, average_precision_score, f1_score, accuracy_score, auc
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from control.config import args


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.n_labels = args.output_dim
        self.confusion_matrix = np.zeros((self.n_labels, self.n_labels))
        self.batch_size = args.batch_size
        self.best_auc = 0
        self.labels_list = [i for i in range(self.n_labels)]

        self.y_true_multi = []
        self.y_pred_multi = []
    
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
        self.y_true_multi = np.concatenate(self.y_true_multi, 0)
        self.y_pred_multi = np.concatenate(self.y_pred_multi, 0)

        self.y_pred_multi = np.nan_to_num(self.y_pred_multi)

        _auc = roc_auc_score (self.y_true_multi, self.y_pred_multi)
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
            
        result = np.round(np.array([_auc, apr, f1]), 3)
        fpr, tpr, thresholds = roc_curve(y_true_multi_array, self.y_pred_multi[:,1], pos_label=1)
        fnr = 1 - tpr 
        tnr = 1 - fpr
        best_threshold = np.argmax(tpr + tnr)
        # print("Best Threshold: ", thresholds[best_threshold])

        if args.show_roc:
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                label="ROC curve (area = %0.2f)" % roc_auc,
            )
            plt.plot([0, 1], [0, 1], color="navy", linewidth=2, linestyle="--")
            plt.xlim([-0.02, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver operating characteristic Curve Image")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig(f"rocauc_{args.project_name}")
        return result, np.round(tpr[best_threshold], 3), np.round(fnr[best_threshold], 3), np.round(tnr[best_threshold],3), np.round(fpr[best_threshold], 3)

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_labels, self.n_labels))
        self.y_true_multi = []
        self.y_pred_multi = []
