from typing import NamedTuple
from collections import deque

import numpy as np
import torch

class ReplayBuffer:
    def __init__(
        self,
        obs_shape,
        act_shape,
        gamma: float,
        batch_size: int,
        buffer_size: int,
        device,
        n_step: int = 1,
        num_stacked_frames: int = 1,
    ):
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        self.n_step = n_step
        self.obs_shape = obs_shape
        if len(obs_shape) == 1:
            self.pixel_obs = False
            obs_dtype = np.float32 
            if num_stacked_frames != 1:
                raise ValueError('Invalid num_stack_frames')
        elif len(obs_shape) == 3:
            self.pixel_obs = True
            obs_dtype = np.uint8
            self.obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
            self.num_stacked_frames = num_stacked_frames
            self.stacked_idxs = np.empty((self.buffer_size, self.num_stacked_frames), dtype=np.int32)
            self.recent_idxs = deque([0] * self.num_stacked_frames, maxlen=self.num_stacked_frames)
        else:
            raise ValueError('Invalid obs_shape')
        self.obses = np.empty((self.buffer_size, *self.obs_shape), dtype=obs_dtype)
        self.acts = np.empty((self.buffer_size, *act_shape), dtype=np.float32)
        self.rews = np.empty((self.buffer_size,), dtype=np.float32)
        self.dones = np.empty((self.buffer_size,), dtype=bool)
        self.discounts = np.empty((self.buffer_size,), dtype=np.float32)

        self.idx = 0
        self.rng = np.random.default_rng()
        self.full = False
        self.n_steps = np.empty((self.buffer_size, ), dtype=np.int32)
        self.recent_rews = deque([], maxlen=self.n_step)
        self.last_discount = 0
    
    def append(self, obs: np.ndarray, act=None, rew=None, discount=None):
        if self.pixel_obs:
            np.copyto(self.obses[self.idx], obs.transpose(2, 0, 1))
            self.recent_idxs.append(self.idx)
            np.copyto(self.stacked_idxs[self.idx], self.recent_idxs)
            if self.full:
                for i in range(1, self.num_stacked_frames):
                    self.stacked_idxs[self.idx_add(i)][:self.num_stacked_frames - i] = self.stacked_idxs[self.idx_add(i)][self.num_stacked_frames - i]
        else:
            np.copyto(self.obses[self.idx], obs)
        if act is not None:
            np.copyto(self.acts[self.idx], act)
            self.recent_rews.append(rew)
            if len(self.recent_rews) == self.n_step:
                self.rews[self.idx_sub(self.n_step - 1)] = sum(x * self.gamma ** i for i, x in enumerate(self.recent_rews))
                self.discounts[self.idx_sub(self.n_step - 1)] = discount * self.gamma ** self.n_step
            self.dones[self.idx] = False
            self.n_steps[self.idx] = self.n_step
            self.last_discount = discount
        else:
            return_ = 0
            for i in range(1, len(self.recent_rews) + 1):
                return_ = return_ * self.gamma + self.recent_rews[len(self.recent_rews) - i]
                self.rews[self.idx_sub(i)] = return_
                self.discounts[self.idx_sub(i)] = self.last_discount * self.gamma ** i
            self.dones[self.idx] = True
            if self.pixel_obs:
                self.recent_idxs.extend([self.idx_add()] * self.num_stacked_frames)
            self.recent_rews.clear()
            for i in range(1, self.n_step):
                j = self.idx_sub(i)
                if self.dones[j]:
                    break
                self.n_steps[j] = i
        
        self.idx = self.idx_add()
        if not self.full and self.idx == 0:
            self.full = True

    def sample(self):
        idxs = np.arange(0, self.buffer_size if self.full else self.idx)
        is_not_done = ~self.dones[:len(idxs)]
        is_not_done[[self.idx_sub(i + 1) for i in range(self.n_step)]] = False
        idxs = idxs[is_not_done]
        idxs = self.rng.choice(idxs, size=self.batch_size, replace=False)
        next_idxs = (idxs + self.n_steps[idxs]) % self.buffer_size
        if self.pixel_obs:
            stacked_obs = self.obses[self.stacked_idxs[idxs]].reshape(self.batch_size, self.num_stacked_frames * self.obs_shape[0], self.obs_shape[1], self.obs_shape[2])
            obs = torch.from_numpy(stacked_obs).to(device=self.device, dtype=torch.float32)
            stacked_next_obs = self.obses[self.stacked_idxs[next_idxs]].reshape(self.batch_size, self.num_stacked_frames * self.obs_shape[0], self.obs_shape[1], self.obs_shape[2])
            next_obs = torch.from_numpy(stacked_next_obs).to(device=self.device, dtype=torch.float32)
        else:
            obs = torch.from_numpy(self.obses[idxs]).to(device=self.device, dtype=torch.float32)
            next_obs = torch.from_numpy(self.obses[next_idxs]).to(device=self.device, dtype=torch.float32)
        act = torch.from_numpy(self.acts[idxs]).to(device=self.device, dtype=torch.float32)
        rew = torch.from_numpy(self.rews[idxs]).to(device=self.device, dtype=torch.float32)
        discount = torch.from_numpy(self.discounts[idxs]).to(device=self.device, dtype=torch.float32)
        return ReplayBufferSamples(obs, act, rew, next_obs, discount)

    def idx_add(self, n=1):
        return (self.idx + n) % self.buffer_size

    def idx_sub(self, n=1):
        return (self.idx + self.buffer_size - n) % self.buffer_size
    
class ReplayBufferSamples(NamedTuple):
    obs: torch.Tensor
    act: torch.Tensor
    rew: torch.Tensor
    next_obs: torch.Tensor
    discount: torch.Tensor