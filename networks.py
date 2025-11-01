import torch
from torch import nn

class DeterministicActor(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    
    def forward(self, obs):
        action = torch.tanh(self.net(obs))
        return action

class DoubleQCritic(nn.Module):
    def __init__(self, q1_net, q2_net, trunk=None):
        super().__init__()
        if trunk is None:
            self.trunk = nn.Identity()
        else:
            self.trunk = trunk
        self.q1_net = q1_net
        self.q2_net = q2_net

    def forward(self, obs, action):
        obs = self.trunk(obs)
        obs_action = torch.cat([obs, action], dim=1)
        return self.q1_net(obs_action).squeeze(1), self.q2_net(obs_action).squeeze(1)