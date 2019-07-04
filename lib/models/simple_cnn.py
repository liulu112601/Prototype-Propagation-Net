import torch
import torch.nn as nn
from .initialization import init_kaiming


def conv_block(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 3, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(),
    nn.MaxPool2d(2)
  )

class Simple_Cnn(nn.Module):

  def __init__(self, x_dim, hid_dim=64, z_dim=64, o_dim=64):
    super(Simple_Cnn, self).__init__()
    self.encoder = nn.Sequential(
      conv_block(x_dim, hid_dim),
      conv_block(hid_dim, hid_dim),
      conv_block(hid_dim, hid_dim),
      conv_block(hid_dim, z_dim),
    )
    self.apply( init_kaiming )

  def forward(self, x, pool=True):
    x = self.encoder(x)
    if pool:
      pool  = x.view(x.size(0), -1) 
      return pool
    else: return x
 

def simpleCnn(x_dim=3):
  model = Simple_Cnn(x_dim=x_dim, hid_dim=64, z_dim=64)
  return model

