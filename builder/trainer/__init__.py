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
from .trainer import *

def get_trainer(args, iteration, x, y, model, logger, device, scheduler, optimizer, criterion, flow_type="train"):
    if args.trainer == "binary_classification": 
        model, iter_loss = binary_classification(args, iteration, x, y, model, logger, device, scheduler, optimizer, criterion, flow_type)
    else:
        print("Selected trainer is not prepared yet...")
        exit(1)

    return model, iter_loss