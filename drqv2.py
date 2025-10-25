from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import dm_env

from replay_buffer import ReplayBuffer
import utils

class DrQV2:
    def __init__(
        self,
        env: dm_env.Environment,
        fe: nn.Module,
        actor: nn.Module,
        critic: nn.Module,
        critic_target: nn.Module,
        fe_lr: float,
        actor_lr: float,
        critic_lr: float,
        tau: float,
        std_schedule: str,
        std_clip: float,
        replay_buffer: ReplayBuffer,
        seed_timesteps: int,
        eval_interval_timesteps: int,
        eval_episodes: int,
        log_dir: Path,
        device,
    ):
        self.env = env
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.critic_target = critic_target.to(device)
        self.replay_buffer = replay_buffer
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.tau = tau
        self.writer = SummaryWriter(log_dir)
        self.seed_timesteps = seed_timesteps
        self.eval_interval_timesteps = eval_interval_timesteps
        self.eval_episodes = eval_episodes
        self.device = device
        self.timesteps = 0
        self.episodes = 0
        self.best_score = -np.inf
        self.best_fe_state_dict = {}
        self.best_actor_state_dict = {}
        self.best_score_timesteps = 0
        self.fe = fe.to(device)
        self.fe_optimizer = Adam(self.fe.parameters(), lr=fe_lr)
        self.aug = RandomShiftsAug(pad=4)
        self.schedule = create_schedule_fn(std_schedule)
        self.std_clip = std_clip
    
    def learn(
        self,
        total_timesteps,
    ):
        act_spec = self.env.action_spec()
        return_ = 0
        timestep = self.env.reset()
        while self.timesteps < total_timesteps:
            obs = timestep.observation
            if timestep.last():
                self.replay_buffer.append(obs[-1])
                self.episodes += 1
                if self.timesteps % self.eval_interval_timesteps == 0:
                    self.eval()
                self.writer.add_scalar('Return', return_, self.timesteps)
                return_ = 0
                timestep = self.env.reset()
                continue
            if self.timesteps < self.seed_timesteps:
                act = np.random.uniform(act_spec.minimum, act_spec.maximum, act_spec.shape)
            else:
                self.update()
                act = self.predict(obs, deterministic=False)
            timestep = self.env.step(act)
            assert timestep.discount == 0 or timestep.discount == 1.0
            self.replay_buffer.append(obs[-1], act, timestep.reward, timestep.discount)
            self.timesteps += 1
            return_ += timestep.reward
            
    def predict(self, obs, deterministic=True):
        obs = obs.transpose(0, 3, 1, 2)
        obs = obs.reshape(-1, obs.shape[2], obs.shape[3])
        obs = torch.from_numpy(obs).to(self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            h = self.fe(obs)
            dist = self.actor(h, self.schedule(self.timesteps))
        act = dist.mean if deterministic else dist.sample()
        return act.cpu().numpy().squeeze(0)
    
    def eval(self):
        if not self.eval_episodes > 0:
            return
        score = utils.evaluate(self.predict, self.env, self.eval_episodes)
        self.writer.add_scalar('Score', score, self.timesteps)
        if score > self.best_score:
            self.best_score = score
            self.best_fe_state_dict = self.fe.state_dict()
            self.best_actor_state_dict = self.actor.state_dict()
            self.best_score_timesteps = self.timesteps

    def update(self):
        replay_data = self.replay_buffer.sample()
        obs_h = self.fe(self.aug(replay_data.obs))
        with torch.no_grad():
            next_obs_h = self.fe(self.aug(replay_data.next_obs))
            std = self.schedule(self.timesteps)
            dist = self.actor(next_obs_h, std)
            next_act = dist.sample(clip=self.std_clip)
            next_target_q1, next_target_q2 = self.critic_target(next_obs_h, next_act)
            next_target_q = torch.min(next_target_q1, next_target_q2)
            next_target_v = next_target_q
            target_q = replay_data.rew + replay_data.discount * next_target_v
        q1, q2 = self.critic(obs_h, replay_data.act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.fe_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.fe_optimizer.step()
        self.critic_optimizer.step()
    
        obs_h = obs_h.detach()
        dist = self.actor(obs_h, std)
        act = dist.sample(clip=self.std_clip)
        q1, q2 = self.critic(obs_h, act)
        q = torch.min(q1, q2)
        actor_loss = torch.mean(-q)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        
        with torch.no_grad():
            for param, param_target in zip(self.critic.parameters(), self.critic_target.parameters()):
                param_target.lerp_(param, self.tau)
        
        self.writer.add_scalar('Actor Loss', actor_loss.item(), self.timesteps)
        self.writer.add_scalar('Critic Loss', critic_loss.item(), self.timesteps)

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

def create_schedule_fn(schdl: str):
    try:
        val = float(schdl)
        return lambda step: val
    except ValueError:
        if schdl.startswith('linear(') and schdl.endswith(')'):
            try:
                parts = schdl[7:-1].split(',')
                if len(parts) == 3:
                    init, final, duration = [float(g) for g in parts]
                    if duration <= 0:
                        return lambda step: final
                    def linear_fn(step: int) -> float:
                        mix = np.clip(step / duration, 0.0, 1.0)
                        return (1.0 - mix) * init + mix * final
                    return linear_fn
            except (ValueError, TypeError):
                pass
        elif schdl.startswith('step_linear(') and schdl.endswith(')'):
            try:
                parts = schdl[12:-1].split(',')
                if len(parts) == 5:
                    init, final1, duration1, final2, duration2 = [
                        float(g) for g in parts
                    ]
                    inv_duration1 = 1.0 / duration1 if duration1 > 0 else 0.0
                    inv_duration2 = 1.0 / duration2 if duration2 > 0 else 0.0
                    def step_linear_fn(step: int) -> float:
                        if step <= duration1:
                            if inv_duration1 == 0.0:
                                return final1
                            mix = np.clip(step * inv_duration1, 0.0, 1.0)
                            return (1.0 - mix) * init + mix * final1
                        else:
                            if inv_duration2 == 0.0:
                                return final2
                            mix = np.clip((step - duration1) * inv_duration2, 0.0, 1.0)
                            return (1.0 - mix) * final1 + mix * final2
                    return step_linear_fn
            except (ValueError, TypeError):
                pass
    raise NotImplementedError(schdl)