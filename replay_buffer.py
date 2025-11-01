from typing import NamedTuple

import numpy as np
import torch
import dm_env.specs

class ReplayBuffer:
    def __init__(self, obs_spec: dm_env.specs.Array, action_spec: dm_env.specs.Array, buffer_size, batch_size, gamma, device, n_step=1, num_stacked_frames=1):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.n_step = n_step
        self.num_stacked_frames = num_stacked_frames
        
        if obs_spec.dtype == np.uint8 and len(obs_spec.shape) == 3:
            self.pixel_obs = True
            self.obs_dtype = np.uint8
            self.obs_shape = (obs_spec.shape[2], obs_spec.shape[0], obs_spec.shape[1])
        else:
            self.pixel_obs = False
            self.obs_dtype = np.float32
            self.obs_shape = obs_spec.shape
        self.obs_buffer = np.empty((self.buffer_size, *self.obs_shape), dtype=self.obs_dtype)

        self.action_dtype = np.float32
        self.action_shape = action_spec.shape
        self.action_buffer = np.empty((self.buffer_size, *self.action_shape), dtype=self.action_dtype)

        self.reward_buffer = np.empty((self.buffer_size,), dtype=np.float32)
        self.discount_buffer = np.empty((self.buffer_size,), dtype=np.float32)
        self.done_buffer = np.empty((self.buffer_size,), dtype=bool)
        
        self.rng = np.random.default_rng() 
        
        self.idx = 0
        self.current_buffer_size = 0
    
    def idx_add(self, i=None, n=1):
        return (i or self.idx + n) % self.buffer_size

    def idx_sub(self, i=None, n=1):
        return (i or self.idx + self.buffer_size - n) % self.buffer_size

    def add(self, obs, action=None, reward=None, discount=None):
        if self.pixel_obs:
            obs = obs.transpose(2, 0 ,1)
        np.copyto(self.obs_buffer[self.idx], obs)
        if action is None:
            self.done_buffer[self.idx] = True
        else:
            self.done_buffer[self.idx] = False
            np.copyto(self.action_buffer[self.idx], action)
            self.reward_buffer[self.idx] = reward
            self.discount_buffer[self.idx] = discount
        
        if self.current_buffer_size != self.buffer_size:
            self.current_buffer_size = self.idx + 1
        self.idx = self.idx_add()
    
    def sample(self):
        obs_shape = (self.num_stacked_frames * self.obs_shape[0], self.obs_shape[1], self.obs_shape[2]) if self.pixel_obs else self.obs_shape
        obs_batch = torch.empty((self.batch_size, *obs_shape), dtype=torch.float32)
        next_obs_batch = torch.empty((self.batch_size, *obs_shape), dtype=torch.float32)
        action_batch = torch.empty((self.batch_size, *self.action_shape), dtype=torch.float32)
        reward_batch = torch.empty((self.batch_size,), dtype=torch.float32)
        discount_batch = torch.empty((self.batch_size,), dtype=torch.float32)

        recent_idxs = {self.idx_sub(n=n) for n in range(self.n_step)}
        for b in range(self.batch_size):
            i = self.rng.integers(self.current_buffer_size)
            while self.done_buffer[i] or i in recent_idxs:
                i = self.rng.integers(self.current_buffer_size)

            if self.pixel_obs:
                frames = np.empty((self.num_stacked_frames, *self.obs_shape))
                frames[self.num_stacked_frames - 1] = self.obs_buffer[i]
                for j in range(1, self.num_stacked_frames):
                    k = self.idx_sub(i, j)
                    s = self.num_stacked_frames - j - 1
                    if self.done_buffer[k]:
                        frames[:s + 1] = frames[s + 1][None, ...]
                        break
                    frames[s] = self.obs_buffer[k]
                obs_batch[b] = torch.from_numpy(frames.reshape((*obs_shape,)).astype(np.float32))
            else:
                obs_batch[b] = torch.from_numpy(self.obs_buffer[i].astype(np.float32))

            action_batch[b] = torch.from_numpy(self.action_buffer[i])

            return_ = 0
            discount = 1
            for j in range(self.n_step):
                t = self.idx_add(i, j)
                if self.done_buffer[t]:
                    t = self.idx_sub(t, 1)
                    break
                return_ += self.reward_buffer[t] * discount
                discount *= self.gamma
            discount *= self.discount_buffer[t]
            reward_batch[b] = float(return_)
            discount_batch[b] = float(discount)

            t = self.idx_add(t, 1)
            if self.pixel_obs:
                frames = np.empty((self.num_stacked_frames, *self.obs_shape))
                frames[self.num_stacked_frames - 1] = self.obs_buffer[t]
                for j in range(1, self.num_stacked_frames):
                    k = self.idx_sub(t, j)
                    s = self.num_stacked_frames - j - 1
                    if self.done_buffer[k]:
                        frames[:s + 1] = frames[s + 1][None, ...]
                        break
                    frames[s] = self.obs_buffer[k]
                next_obs_batch[b] = torch.from_numpy(frames.reshape((*obs_shape,)).astype(np.float32))
            else:
                next_obs_batch[b] = torch.from_numpy(self.obs_buffer[t].astype(np.float32))
    
        return ReplayBufferSamples(obs_batch.to(self.device), action_batch.to(self.device), reward_batch.to(self.device), next_obs_batch.to(self.device), discount_batch.to(self.device))

class ReplayBufferSamples(NamedTuple):
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    discount: torch.Tensor