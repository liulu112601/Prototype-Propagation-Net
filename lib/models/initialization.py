import torch
import torch.nn as nn


def init_resnet(m):
  if isinstance(m, nn.Conv2d):
    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
  elif isinstance(m, nn.BatchNorm2d):
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)
  elif isinstance(m, nn.Linear):
    nn.init.normal_(m.weight, 0, 0.01)
    if m.bias is not None: nn.init.constant_(m.bias, 0)

def init_kaiming(m):
  if isinstance(m, nn.Conv2d):
    nn.init.kaiming_normal_(m.weight)
  elif isinstance(m, nn.BatchNorm2d):
    nn.init.constant_(m.weight, 1)
    if m.bias is not None: nn.init.constant_(m.bias, 0)
  elif isinstance(m, nn.Linear):
    nn.init.normal_(m.weight, 0, 0.01)
    if m.bias is not None: nn.init.constant_(m.bias, 0)
