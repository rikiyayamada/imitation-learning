from copy import deepcopy

import torch
from torch import nn, optim
import numpy as np

from . import DeterministicActor, DoubleQCritic, Encoder

class LinearDecayGaussianNoise:
    def __init__(self, init_std, final_std, duration, low=-1.0, high=1.0, eps=1e-6, clip: float | None = None):
        self.init_std = init_std
        self.final_std = final_std
        self.duration = duration
        self.low = low
        self.high = high
        self.eps = eps
        self.clip = clip
    
    def __call__(self, action, step, clip=False):
        mix = np.clip(step / self.duration, 0.0, 1.0)
        std = (1.0 - mix) * self.init_std + mix * self.final_std
        std = torch.ones_like(action) * std
        eps = torch.randn_like(action)
        eps *= std
        if clip:
            eps = torch.clamp(eps, -self.clip, self.clip)
        action = action + eps
        clamped_action = torch.clamp(action, self.low + self.eps, self.high - self.eps)
        action = action - action.detach() + clamped_action.detach()
        return action

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = nn.functional.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return nn.functional.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

class DrQv2:
    def __init__(
        self,
        actor: DeterministicActor,
        critic: DoubleQCritic,
        adam_kwargs: dict,
        tau: float,
        noise: LinearDecayGaussianNoise,
        device,
        encoder: Encoder | None = None,
        aug: RandomShiftsAug | None = None,
    ):
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.target_critic = deepcopy(critic).requires_grad_(False).eval().to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), **adam_kwargs)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), **adam_kwargs)
        self.tau = tau
        self.noise = noise
        self.encoder = (encoder or nn.Identity()).to(self.device)
        self.critic_optimizer.add_param_group({'params': self.encoder.parameters()})
        self.aug = (aug or nn.Identity()).to(self.device)
    
    def update_critic(self, obs, action, reward, next_obs, discount, additional_state, next_additional_state, step):
        log_info = {}
        with torch.no_grad():
            next_action = self.noise(self.actor(next_obs, next_additional_state), step, clip=True)
            next_target_q1, next_target_q2 = self.target_critic(next_obs, next_action, next_additional_state)
            next_target_q = torch.min(next_target_q1, next_target_q2)
            next_target_v = next_target_q
            target_q = reward + discount * next_target_v
        q1, q2 = self.critic(obs, action, additional_state)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        log_info['critic_loss'] = critic_loss.item()
        return log_info
    
    def update_action(self, obs, additional_state, step):
        log_info = {}
        action = self.noise(self.actor(obs, additional_state), step, clip=True)
        q1, q2 = self.critic(obs, action, additional_state)
        q = torch.min(q1, q2)
        actor_loss = torch.mean(-q)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        log_info['actor_loss'] = actor_loss.item()
        return log_info
    
    def update_target_critic(self):
        with torch.no_grad():
            for param, param_target in zip(self.critic.parameters(), self.target_critic.parameters()):
                param_target.lerp_(param, self.tau)

    def update(
        self,
        obs,
        action,
        reward,
        next_obs,
        discount,
        step,
        additional_state=None,
        next_additional_state=None
    ):
        log_info = {}
        obs = self.aug(obs)
        h = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.aug(next_obs)
            next_h = self.encoder(next_obs)
        log_info |= self.update_critic(h, action, reward, next_h, discount, additional_state, next_additional_state, step)
        h = h.detach()
        log_info |= self.update_action(h, additional_state, step)
        self.update_target_critic()
        return log_info

    def predict(self, obs, step=None, additional_state=None):
        obs = torch.from_numpy(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            h = self.encoder(obs)
            action = self.actor(h, additional_state)
        if step is not None:
            action = self.noise(action, step)
        return action.squeeze(0).cpu().numpy()