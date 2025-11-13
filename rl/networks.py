import torch
from torch import nn

class DeterministicActor(nn.Module):
    def __init__(self, net, trunk=None):
        super().__init__()
        self.trunk = trunk or nn.Identity()
        self.net = net
    
    def forward(self, obs, add_state=None):
        h = self.trunk(obs)
        if add_state is not None:
            h = torch.cat([h, add_state], dim=1)
        action = torch.tanh(self.net(h))
        return action

class DoubleQCritic(nn.Module):
    def __init__(self, q1_net, q2_net, trunk=None):
        super().__init__()
        self.trunk = trunk or nn.Identity()
        self.q1_net = q1_net
        self.q2_net = q2_net

    def forward(self, obs, action, add_state=None):
        h = self.trunk(obs)
        if add_state is not None:
            h = torch.cat([h, add_state], dim=1)
        h_action = torch.cat([h, action], dim=1)
        q1, q2 = self.q1_net(h_action).squeeze(1), self.q2_net(h_action).squeeze(1)
        return q1, q2

class Encoder(nn.Module):
    def __init__(self, convnet):
        super().__init__()
        self.convnet = convnet
    
    def forward(self, obs):
        obs = obs / 256.0 - 0.5
        h = self.convnet(obs)
        return h