import os
import yaml
import argparse
import torch

seed_list = [0, 1004, 9209, 909, 30, 31, 2021, 2022]

### CONFIGURATIONS
parser = argparse.ArgumentParser()

# General Parameters
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=int, default=1, nargs='+')
parser.add_argument('--cpu', default=False, action='store_true')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--reset', default=False, action='store_true')
parser.add_argument('--project-name', type=str, default="test")
parser.add_argument('--checkpoint', '-cp', type=bool, default=False)

# Training Parameters
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--l2-coeff', type=float, default=0.002)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--activation', help='activation function of the networks', choices=['selu','relu'], default='relu', type=str) #invase
parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'sgd_lars','adam', 'adam_lars','adamw', 'adamw_lars'])
parser.add_argument('--lr-scheduler', type=str, default="Single" , choices=["CosineAnnealing", "Single"])
parser.add_argument('--lr-init', type=float, default=1e-3) # not being used for CosineAnnealingWarmUpRestarts...
parser.add_argument('--lr_max', type=float, default=4e-3)
parser.add_argument('--t_0', '-tz', type=int, default=5, help='T_0 of cosine annealing scheduler')
parser.add_argument('--t_mult', '-tm', type=int, default=2, help='T_mult of cosine annealing scheduler')
parser.add_argument('--t_up', '-tup', type=int, default=1, help='T_up (warm up epochs) of cosine annealing scheduler')
parser.add_argument('--gamma', '-gam', type=float, default=0.5, help='T_up (warm up epochs) of cosine annealing scheduler')
parser.add_argument('--momentum', '-mo', type=float, default=0.9, help='Momentum of optimizer')
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-6, help='Weight decay of optimizer')
parser.add_argument('--patient-time', default=False)
parser.add_argument('--threshold', type=float, default=0.5)


# Data Parameters
parser.add_argument('--k-cross-fold-validation', type=int, default=4)
parser.add_argument('--val-data-ratio', type=float, default=0.2)
parser.add_argument('--test-data-ratio', type=float, default=0.2)
parser.add_argument('--data-type', type=str, default="hourly", choices=["hourly", "signal"])
parser.add_argument('--model-type', type=str, default="1")

# ECG Data Parameters
parser.add_argument('--prediction-after', type=int, default=24, choices=[24, 72], help='prediction vasso after x hours')
parser.add_argument('--window-size', type=int, default=1, help='unit is second')
parser.add_argument('--resp-min-size', type=int, default=1, help='unit is second')

# Model Parameters
parser.add_argument('--trainer', type=str, default="binary_classification")
parser.add_argument('--model', type=str, default="ecg_resp_vs_v1") #model name
parser.add_argument('--enc-model', type=str, default="raw", choices= ['psd', 'sincnet', 'lfcc', 'raw'])

# Visualize / Logging Parameters
parser.add_argument('--log-iter', type=int, default=10)

# Test / Store Parameters
parser.add_argument('--best', default=True, action='store_true')
parser.add_argument('--last', default=False, action='store_true')

args = parser.parse_args()
args.output_dim = 2

# Dataset Path settings
with open('./control/path_configs.yaml') as f:
    path_configs = yaml.safe_load(f)
    args.data_path = path_configs['data_directory']['data_path']
    args.dir_root =  os.getcwd()
    args.dir_result = path_configs['dir_result']
