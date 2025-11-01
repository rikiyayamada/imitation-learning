from copy import deepcopy

import numpy as np
import torch
from torch import nn, optim

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

class TDPG:
    def __init__(
        self,
        encoder: nn.Module,
        actor: nn.Module,
        critic: nn.Module,
        adam_kwargs: dict,
        tau: float,
        noise: LinearDecayGaussianNoise,
        device,
    ):
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() and device != 'cpu' else 'cpu'
        self.encoder = encoder.to(self.device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.target_critic = deepcopy(critic).requires_grad_(False).eval().to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), **adam_kwargs)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), **adam_kwargs)
        self.critic_optimizer.add_param_group({'params': self.encoder.parameters()})
        self.tau = tau
        self.noise = noise
    
    def update(self, obs, action, rew, next_obs, discount, step):
        log_info = {}
        h = self.encoder(obs)

        # update critic
        with torch.no_grad():
            next_h = self.encoder(next_obs)
            next_action = self.noise(self.actor(next_h), step, clip=True)
            next_target_q1, next_target_q2 = self.target_critic(next_h, next_action)
            next_target_q = torch.min(next_target_q1, next_target_q2)
            next_target_v = next_target_q
            target_q = rew + discount * next_target_v
        q1, q2 = self.critic(h, action)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        log_info['critic_loss'] = critic_loss.item()
    
        # update actor
        h = h.detach()
        action = self.noise(self.actor(h), step, clip=True)
        q1, q2 = self.critic(h, action)
        q = torch.min(q1, q2)
        actor_loss = torch.mean(-q)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        log_info['actor_loss'] = actor_loss.item()
    
        # update target_critic
        with torch.no_grad():
            for param, param_target in zip(self.critic.parameters(), self.target_critic.parameters()):
                param_target.lerp_(param, self.tau)
        
        return log_info

    def predict(self, obs, step, deterministic):
        obs = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            h = self.encoder(obs)
            action = self.actor(h)
        if not deterministic:
            action = self.noise(action, step)
        return action.cpu().numpy().squeeze(0)