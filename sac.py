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

class SAC:
    def __init__(
        self,
        env: dm_env.Environment,
        actor: nn.Module,
        critic: nn.Module,
        critic_target: nn.Module,
        alpha: float,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        tau: float,
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
        self.best_state_dict = {}
        self.best_score_timesteps = 0
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, device=device, requires_grad=True)
        self.log_alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -env.action_spec().shape[0]

    def learn(
        self,
        total_timesteps,
    ):
        act_spec = self.env.action_spec()
        return_ = 0
        timestep = self.env.reset()
        while self.timesteps < total_timesteps:
            if timestep.last():
                self.replay_buffer.append(timestep.observation)
                self.episodes += 1
                if self.timesteps % self.eval_interval_timesteps == 0:
                    self.eval()
                self.writer.add_scalar('Return', return_, self.timesteps)
                return_ = 0
                timestep = self.env.reset()
                continue
            obs = timestep.observation
            if self.timesteps < self.seed_timesteps:
                act = np.random.uniform(act_spec.minimum, act_spec.maximum, act_spec.shape)
            else:
                self.update()
                act = self.actor.predict(obs, deterministic=False)
            timestep = self.env.step(act)
            assert timestep.discount == 0 or timestep.discount == 1.0
            self.replay_buffer.append(obs, act, timestep.reward, timestep.discount)
            self.timesteps += 1
            return_ += timestep.reward

    def eval(self):
        score = utils.evaluate(self.actor.predict, self.env, self.eval_episodes)
        self.writer.add_scalar('Score', score, self.timesteps)
        if score > self.best_score:
            self.best_score = score
            self.best_state_dict = self.actor.state_dict()
            self.best_score_timesteps = self.timesteps
    
    def update(self):
        replay_data = self.replay_buffer.sample()
        with torch.no_grad():
            dist = self.actor(replay_data.next_obs)
            next_act, next_log_prob = dist.act_log_prob()
            next_target_q1, next_target_q2 = self.critic_target(replay_data.next_obs, next_act)
            next_target_q = torch.min(next_target_q1, next_target_q2)
            next_target_v = next_target_q - self.log_alpha.exp() * next_log_prob
            target_q = replay_data.rew + replay_data.discount * next_target_v
        q1, q2 = self.critic(replay_data.obs, replay_data.act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
    
        dist = self.actor(replay_data.obs)
        act, log_prob = dist.act_log_prob()
        q1, q2 = self.critic(replay_data.obs, act)
        q = torch.min(q1, q2)
        actor_loss = torch.mean(self.log_alpha.exp().detach() * log_prob - q)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        
        alpha_loss = torch.mean(self.log_alpha.exp() * (-log_prob - self.target_entropy).detach())
        self.log_alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        
        with torch.no_grad():
            for param, param_target in zip(self.critic.parameters(), self.critic_target.parameters()):
                param_target.lerp_(param, self.tau)
        
        self.writer.add_scalar('Actor Loss', actor_loss.item(), self.timesteps)
        self.writer.add_scalar('Critic Loss', critic_loss.item(), self.timesteps)
        self.writer.add_scalar('Alpha Loss', alpha_loss.item(), self.timesteps)