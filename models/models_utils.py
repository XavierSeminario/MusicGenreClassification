import torch.nn as nn
import torch.nn.functional as F
import torch


def init_weights(self, method = 'Kaiming'):
    for m in self.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            if method == 'Kaiming':
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)      
