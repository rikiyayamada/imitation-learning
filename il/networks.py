import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, net, trunk=None):
        super().__init__()
        self.trunk = trunk or nn.Identity()
        self.net = net

    def forward(self, obs, act, additional_state=None):
        h = self.trunk(obs)
        if additional_state is not None:
            h = torch.cat([h, additional_state], dim=1)
        h_act = torch.cat([h, act], dim=1)
        return self.net(h_act).squeeze(1)