"""
    - Unit testing code for generator and discriminator
    - TODO : add code for classifier unit testing
"""  
import torch
import torchvision
from torch import nn
from models import discriminator, global_discriminator, local_discriminator
from models import generator


# discriminator
def d_test():
  x = torch.randn(1,3,128,128)
  y = torch.randn(1,3,256,256)
  model = ContextDiscriminator((3,128,128),(3,256,256),arc=2)
  preds = model((x,y))
  print(preds) 

# generator
def test():
    x=torch.randn(1,3,256,256)
    model=Generator(3,3,1)
    preds=model(x)
    print(preds.shape)    