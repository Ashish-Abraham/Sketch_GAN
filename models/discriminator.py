import torch
import torchvision
from torch import nn
from .local_discriminator import LocalDiscriminator
from .global_discriminator import GlobalDiscriminator

class ContextDiscriminator(nn.Module):
    def __init__(self, local_input_shape, global_input_shape, arc=None):
        super(ContextDiscriminator, self).__init__()
        self.arc = arc
        self.input_shape = [local_input_shape, global_input_shape]
        self.output_shape = (1,)
        self.model_ld = LocalDiscriminator(local_input_shape)
        self.model_gd = GlobalDiscriminator(global_input_shape, arc=arc)
        # input_shape: [(None, 1024), (None, 1024)]
        in_features = self.model_ld.output_shape[-1] + self.model_gd.output_shape[-1]
        self.concat1 = Concatenate(dim=-1)
        # input_shape: (None, 2048)
        self.linear1 = nn.Linear(in_features, 1)
        # self.act1 = nn.ReLU()
        self.act2 = nn.Tanh()
        # output_shape: (None, 1)

    def forward(self, x):
        x_ld, x_gd = x
        x_ld = self.model_ld(x_ld)
        x_gd = self.model_gd(x_gd)
        out = self.act2(self.linear1(self.concat1([x_ld, x_gd])))
        return out