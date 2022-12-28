import torch
import torchvision
from torch import nn
from .modules.generator_modules import DownSampleConv, UpSampleConv, GenMod

class Generator(nn.Module):
  
  def __init__(self,in_channels,out_channels,batch_size):
    """
        - 3 stage cascaded architecture
        - Combining outout of previous stages with corrupted image and input to next stage
    """  
    super().__init__()
    self.Stage1 = GenMod(3,3)
    self.Stage2 = GenMod(3,3)
    self.Stage3 = GenMod(3,3)
    self.batch_size = batch_size
  
  def forward(self,x):
    y = self.Stage1(x)
    y = torch.add(y,x)
    y = nn.functional.normalize(y, p=1.0, dim = 0)
    z = self.Stage2(y)
    z = torch.add(z,y)
    z = nn.functional.normalize(z, p=1.0, dim = 0)
    w = self.Stage3(z)
    return w
