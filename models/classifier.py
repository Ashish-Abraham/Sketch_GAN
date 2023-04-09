import torch.nn as nn
import torchvision
from torch.utils.data import Dataloader, Dataset, random_split

class Block1(nn.Module):
    def __init__(self,in_channels):
        super(Block1, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=15, stride=3, padding=0),
            nn.ReLU(inplace=True)
        )
        self.pool_layer1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.pool_layer2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.pool_layer1(x)
        x = self.conv_layer2(x)
        x = self.pool_layer2(x)
        return x
    
class Block2(nn.Module):
    def __init__(self):
        super(Block2,self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2D(in_channels=128, out_channels=256, kernal_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2D(in_channels=256, out_channels=256, kernal_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2D(in_channels=256, out_channels=256, kernal_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool_layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.pool_layer(x)
        return x

class Block3(nn.Module):
    def __init__(self):
        super(Block3, self).__init__()
        self.conv_layer1 = nn.Sequential(
          nn.Conv2d(in_channels=256, out_channels=512, kernel_size=7, stride=1, padding=0),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5),
        )
        self.conv_layer2 = nn.Sequential(
          nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5)
        )
        
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        return x


class Sketch_A_Net(nn.Module):
    def __init__(self,in_channels):
        super(Sketch_A_Net,self).__init__()
        # L1,L2 in paper
        self.Layer1 = Block1(in_channels)
        # L3, L4, L5 in paper
        self.Layer2 = Block2()
        # L6, L7 in paper
        self.Layer3 = Block3()
        # Final layer
        self.final_conv = nn.Conv2D(in_channels=512, out_channels=250, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = self.final_conv(x)
        return x