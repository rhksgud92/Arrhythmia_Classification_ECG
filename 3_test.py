import os
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import random

import torch
from torch.autograd import Variable
import torch.nn as nn

from control.config import args
from builder.data.data_realtime_preprocess import get_data_preprocess_test
from builder.models import get_model
from builder.utils.metrics import Evaluator
from builder.utils.logger import Logger
from builder.trainer.trainer import *
from builder.utils.utils import set_seeds, set_devices
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
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
print(device)
# seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)

test_loader = get_data_preprocess_test(args)
model = get_model(args)
model = model(args, device).to(device)
name = args.project_name

# Check if checkpoint exists
if args.last:
    ckpt_path = args.dir_result + '/' + name + '/ckpts/best.pth'
elif args.best:
    ckpt_path = args.dir_result + '/' + name + '/ckpts/best.pth'

if not os.path.exists(ckpt_path):
    print("Error write correct test dataset path...")
    exit(1)

ckpt = torch.load(ckpt_path, map_location=device)

state = {k: v for k, v in ckpt['model'].items()}
model.load_state_dict(state)

model.eval()
print('model loaded')

logger.evaluator.reset()
result_list = []
with torch.no_grad():
    for idx, batch in enumerate(tqdm(test_loader)):
        test_x, test_y = batch
        test_x = test_x.to(device)
        test_y = test_y.to(device)

        model, _ = binary_classification(args, iteration, test_x, test_y, model, logger, device, scheduler,
                                            optimizer, criterion, flow_type="test")   

logger.test_result_only()

        