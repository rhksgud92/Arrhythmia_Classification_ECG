import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import importlib
from torch import Tensor
from typing import Callable, Any, Optional, List

class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (self.stride == 1) and (inp == oup)

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])            
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:

        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        

class CNN2D_RESNET_V3_SMALL2(nn.Module):
        def __init__(self, args, device):
                super(CNN2D_RESNET_V3_SMALL2, self).__init__()      
                self.args = args
                self.dropout = 0.1
                self.feature_extractor = args.enc_model

                activation = 'relu'
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()],
                        ['relu', nn.ReLU(inplace=True)],
                        ['tanh', nn.Tanh()],
                        ['sigmoid', nn.Sigmoid()],
                        ['leaky_relu', nn.LeakyReLU(0.2)],
                        ['elu', nn.ELU()]
                ])
                def conv2d_bn(inp, oup, kernel_size, stride, padding):
                        return nn.Sequential(
                                nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding),
                                nn.BatchNorm2d(oup),
                                self.activations[activation],
                                nn.Dropout(self.dropout),
                        )
                
                self.features1 = nn.Sequential(
                        conv2d_bn(1,  8, (1,51), (1,1), (0,25)), 
                        nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),
                        )

                features2 = []
                block = InvertedResidual
                features2.append(block(8, 8, 1, expand_ratio=4))
                features2.append(block(8, 32, 2, expand_ratio=2))
                features2.append(block(32, 32, 1, expand_ratio=4))

                self.features_mobile_resnet = nn.Sequential(*features2)

                self.agvpool = nn.AdaptiveAvgPool2d((8,1))

                self.classifier = nn.Sequential(
                        nn.Linear(in_features=256, out_features= 32, bias=True),
                        nn.BatchNorm1d(32),
                        self.activations[activation],
                        nn.Linear(in_features=32, out_features= args.output_dim, bias=True),
                )


        
        def forward(self, x):
                x = x.unsqueeze(1)
                x = self.features1(x)
                x = self.features_mobile_resnet(x)
                
                x = self.agvpool(x)
                x = x.reshape(x.size(0), -1)
                output = self.classifier(x)
                return output
                
                
        def init_state(self, device):
                pass
         