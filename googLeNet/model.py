import torch.nn as nn
import torch

import torch.nn as nn
import torch

class Inception(nn.Module):
    def __init__(self, in_channel, br_1x1,
                br_3x3_reduce, br_3x3,
                br_5x5_reduce, br_5x5, pool_proj):
        super(Inception, self).__init__()
        self.add_module(
            '1x1', nn.Conv2d(in_channel, br_1x1, kernel_size=1)
        )
        self.add_module('1x1_bn', nn.BatchNorm2d(br_1x1))
        self.add_module('relu_1x1', nn.ReLU())

        
        self.add_module(
            '3x3_reduce',
            nn.Conv2d(in_channel, br_3x3_reduce, kernel_size=1)
        )
        self.add_module('relu_3x3_reduce', nn.ReLU())
        self.add_module(
            '3x3',
            nn.Conv2d(br_3x3_reduce, br_3x3, kernel_size=3, padding=1)
        )
        self.add_module('3x3_bn', nn.BatchNorm2d(br_3x3))
        self.add_module('relu_3x3', nn.ReLU())
        
        self.add_module(
            '5x5_reduce',
            nn.Conv2d(in_channel, br_5x5_reduce, kernel_size=1)
        )
        self.add_module('relu_5x5_reduce', nn.ReLU())
        self.add_module(
            '5x5',
            nn.Conv2d(br_5x5_reduce, br_5x5, kernel_size=5, padding=2)
        )
        self.add_module('5x5_bn', nn.BatchNorm2d(br_5x5))
        self.add_module('relu_5x5', nn.ReLU())
        
        self.add_module(
            'pool',
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.add_module(
            'pool_proj',
            nn.Conv2d(in_channel, pool_proj, kernel_size=1)
        )
        self.add_module('relu_pool_proj', nn.ReLU())

    def forward(self, x):
        x1 = getattr(self, '1x1')(x)
        x1 = getattr(self, '1x1_bn')(x1)
        x1 = getattr(self, 'relu_1x1')(x1)
        
        
        x2 = getattr(self, '3x3_reduce')(x)
        x2 = getattr(self, 'relu_3x3_reduce')(x2)
        x2 = getattr(self, '3x3')(x2)
        x2 = getattr(self, '3x3_bn')(x2)
        x2 = getattr(self, 'relu_3x3')(x2)

        x3 = getattr(self, '5x5_reduce')(x)
        x3 = getattr(self, 'relu_5x5_reduce')(x3)
        x3 = getattr(self, '5x5')(x3)
        x3 = getattr(self, '5x5_bn')(x3)        
        x3 = getattr(self, 'relu_5x5')(x3)

        x4 = getattr(self, 'pool')(x)
        x4 = getattr(self, 'pool_proj')(x4)
        x4 = getattr(self, 'relu_pool_proj')(x4)

        return torch.cat([x1, x2, x3, x4], dim=1)


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.add_module(
            'conv1/7x7_s2',
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        )
        self.add_module('conv1/7x7_s2_bn', nn.BatchNorm2d(64))
        self.add_module('conv1/relu_7x7', nn.ReLU())
        
        self.add_module(
            'pool1', nn.MaxPool2d(3, stride=2, ceil_mode=True)
        )
        
        self.add_module('norm1', nn.LocalResponseNorm(5))
        
        self.add_module('conv2/3x3_reduce', nn.Conv2d(64, 64, kernel_size=1))
        self.add_module('conv2/relu_3x3_reduce', nn.ReLU())
        self.add_module(
            'conv2/3x3',
            nn.Conv2d(64, 192, kernel_size=3, padding=1)
        )
        self.add_module('conv2/3x3_bn', nn.BatchNorm2d(192))
        self.add_module('conv2/relu_3x3', nn.ReLU())
        
        self.add_module('norm2', nn.LocalResponseNorm(5))
        
        self.add_module(
            'pool2',nn.MaxPool2d(3, stride=2, ceil_mode=True)
        )
        
        self.inception_3a = Inception(
            192, 64, 96, 128, 16, 32, 32
        )
        self.inception_3b = Inception(
            256, 128, 128, 192, 32, 96, 64
        )
        
        self.add_module(
            'pool3', nn.MaxPool2d(3, stride=2, ceil_mode=True)
        )
        
        self.inception_4a = Inception(
            480, 192, 96, 208, 16, 48, 64
        )
        self.inception_4b = Inception(
            512, 160, 112, 224, 24, 64, 64
        )
        self.inception_4c = Inception(
            512, 128, 128, 256, 24, 64, 64
        )
        self.inception_4d = Inception(
            512, 112, 144, 288, 32, 64, 64
        )
        self.inception_4e = Inception(
            528, 256, 160, 320, 32, 128, 128
        )
        
        self.add_module(
            'pool4', nn.MaxPool2d(3, stride=2, ceil_mode=True)
        )
        
        self.inception_5a = Inception(
            832, 256, 160, 320, 32, 128, 128
        )
        self.inception_5b = Inception(
            832, 384, 192, 384, 48, 128, 128
        )

        self.add_module('pool5', nn.AvgPool2d(1, stride=1))
    
        self.add_module('drop', nn.Dropout2d(p=0.4))
        
        self.add_module(
            'loss3/classifier', 
            nn.Sequential(
                nn.Linear(1024, 1000),
                nn.ReLU(),
                nn.Linear(1000,128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
            
        )
        
        
    def forward(self, x):
        x = getattr(self, 'conv1/7x7_s2')(x)
        x = getattr(self, 'conv1/7x7_s2_bn')(x)
        x = getattr(self, 'conv1/relu_7x7')(x)
        x = self.pool1(x)
        x = self.norm1(x)
        
        x = getattr(self, 'conv2/3x3_reduce')(x)
        x = getattr(self, 'conv2/relu_3x3_reduce')(x)
        x = getattr(self, 'conv2/3x3')(x)
        x = getattr(self, 'conv2/3x3_bn')(x)
        x = getattr(self, 'conv2/relu_3x3')(x)
        x = self.norm2(x)
        x = self.pool2(x)
        
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.pool3(x)
        
        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        x = self.pool4(x)
        
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.pool5(x)
        x = self.drop(x)
        
        x = x.view(-1, 1024)
        x = getattr(self, 'loss3/classifier')(x)
        # x = self.fc(x)
        return x