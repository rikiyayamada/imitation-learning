import torch
from torch import optim, nn
from torch import autograd

from rl import DrQv2
from . import Discriminator, ReplayBuffer, ExpertReplayBuffer

class DAC:
    def __init__(
        self,
        rl_agent: DrQv2,
        discriminator: Discriminator,
        adam_kwargs: dict,
        replay_buffer: ReplayBuffer,
        demo_path,
        update_rl_agent_every: int,
    ):
        self.rl_agent = rl_agent
        self.device = rl_agent.device
        self.discriminator = discriminator.to(self.device)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), **adam_kwargs)
        self.replay_buffer = replay_buffer
        self.expert_replay_buffer = ExpertReplayBuffer(
            demo_path,
            replay_buffer.batch_size,
            replay_buffer.gamma,
            replay_buffer.obs_shape,
            replay_buffer.n_step,
            replay_buffer.num_stack,
            replay_buffer.device,
            replay_buffer.pixel_obs,
        )
        self.update_rl_agent_every = update_rl_agent_every
        self.batch_size = replay_buffer.batch_size
        self.lambda_term = 10
    
    def compute_reward(self, obs, action, is_absorbing):
        batch_size, n_step = obs.shape[:2]
        flat_obs = obs.view(batch_size * n_step, *obs.shape[2:])
        flat_action = action.view(batch_size * n_step, *action.shape[2:])
        flat_is_absorbing = is_absorbing.view(batch_size * n_step, *is_absorbing.shape[2:])
        with torch.no_grad():
            h = self.rl_agent.encoder(flat_obs)
            h = self.rl_agent.actor.trunk(h)
        flat_reward = self.discriminator(h, flat_action, flat_is_absorbing)
        reward = flat_reward.view(batch_size, n_step)
        discounted_return = torch.zeros(batch_size, device=self.device)
        for i in reversed(range(n_step)):
            discounted_return = reward[:, i] + self.replay_buffer.gamma * discounted_return
        return discounted_return
    
    def update_rl_agent(self, obs, action, is_absorbing, next_obs, next_is_absorbing, discount, step):
        log_info = {}
        with torch.no_grad():
            reward = self.compute_reward(obs, action, is_absorbing)
        log_info |= self.rl_agent.update(obs[:, 0], action[:, 0], reward, next_obs, discount, step, is_absorbing[:, 0], next_is_absorbing)
        return log_info
    
    def compute_gradient_penalty(self, obs, action, is_absorbing, expert_obs, expert_action, expert_is_absorbing):
        valid_mask = (is_absorbing == 0.0) & (expert_is_absorbing == 0.0)
        num_valid_samples = valid_mask.sum().item()
        if num_valid_samples == 0:
            return torch.tensor(0.0, device=self.device)
        mask_indices = valid_mask.squeeze(1)
        obs = obs[mask_indices]
        action = action[mask_indices]
        expert_obs = expert_obs[mask_indices]
        expert_action = expert_action[mask_indices]

        with torch.no_grad():
            h = self.rl_agent.actor.trunk(self.rl_agent.encoder(obs))
            expert_h = self.rl_agent.actor.trunk(self.rl_agent.encoder(expert_obs))
        alpha = torch.rand(num_valid_samples, 1).to(self.device)
        alpha_h = alpha.expand_as(h)
        alpha_action = alpha.expand_as(action)
        interpolated_h = (alpha_h * h + (1 - alpha_h) * expert_h).requires_grad_(True)
        interpolated_action = (alpha_action * action + (1 - alpha_action) * expert_action).requires_grad_(True)
        interpolated_is_absorbing = torch.zeros(num_valid_samples, 1, device=self.device, requires_grad=True)
        d = self.discriminator(interpolated_h, interpolated_action, interpolated_is_absorbing)
        grad = autograd.grad(
            outputs=d,
            inputs=(interpolated_h, interpolated_action),
            grad_outputs=torch.ones_like(d, device=self.device),
            create_graph=True,
            retain_graph=True,
        )
        grad_concat = torch.cat((grad[0].view(grad[0].size(0), -1), grad[1].view(grad[1].size(0), -1)), dim=1)
        grad_penalty = ((grad_concat.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty
    
    def update_discriminator(self, obs, action, is_absorbing):
        log_info = {}
        expert_obs, expert_action, expert_is_absorbing = self.expert_replay_buffer.sample()
        ones = torch.ones(self.batch_size, device=self.device)
        zeros = torch.zeros(self.batch_size, device=self.device)
        label = torch.cat([zeros, ones])
        logit = torch.cat([self.compute_reward(obs, action, is_absorbing), self.compute_reward(expert_obs, expert_action, expert_is_absorbing)])
        grad_penalty = self.compute_gradient_penalty(obs[:, 0], action[:, 0], is_absorbing[:, 0], expert_obs[:, 0], expert_action[:, 0], expert_is_absorbing[:, 0])
        discriminator_loss = nn.functional.binary_cross_entropy_with_logits(logit, label, reduction='mean') + grad_penalty
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        log_info['discriminator_loss'] = discriminator_loss.item()
        return log_info
    
    def update(self, step):
        log_info = {}
        obs, action, is_absorbing, next_obs, next_is_absorbing, discount = self.replay_buffer.sample()
        log_info |= self.update_discriminator(obs, action, is_absorbing)
        if step % self.update_rl_agent_every == 0:
            log_info |= self.update_rl_agent(obs, action, is_absorbing, next_obs, next_is_absorbing, discount, step)
        return log_info
    
    def predict(self, obs, step=None):
        return self.rl_agent.predict(obs, step, additional_state=torch.zeros((1, 1), dtype=torch.float32, device=self.device))