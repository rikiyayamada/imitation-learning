import numpy as np
import torch
from typing import NamedTuple

class ReplayBuffer:
    def __init__(
        self,
        obs_shape,
        act_shape,
        batch_size: int,
        buffer_size: int,
        device,
    ):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        self.obses = np.empty((self.buffer_size, *obs_shape), dtype=np.float32)
        self.acts = np.empty((self.buffer_size, *act_shape), dtype=np.float32)
        self.rews = np.empty((self.buffer_size,), dtype=np.float32)
        self.dones = np.empty((self.buffer_size,), dtype=bool)
        self.discounts = np.empty((self.buffer_size,), dtype=np.float32)

        self.idx = 0
        self.rng = np.random.default_rng()
        self.full = False
    
    def append(self, obs: np.ndarray, act=None, rew=None, discount=None):
        np.copyto(self.obses[self.idx], obs)
        if act is not None:
            np.copyto(self.acts[self.idx], act)
            self.rews[self.idx] = rew
            self.discounts[self.idx] = discount
            self.dones[self.idx] = False
        else:
            self.dones[self.idx] = True
        
        self.idx = (self.idx + 1) % self.buffer_size
        if self.idx == 0:
            self.full = True

    def sample(self):
        idxs = np.arange(0, self.buffer_size if self.full else self.idx)
        is_not_done = ~self.dones[:len(idxs)]
        is_not_done[(self.idx - 1 + self.buffer_size) % self.buffer_size] = False
        idxs = idxs[is_not_done]
        idxs = self.rng.choice(idxs, size=self.batch_size, replace=False)
        obs = torch.from_numpy(self.obses[idxs]).to(device=self.device)
        act = torch.from_numpy(self.acts[idxs]).to(device=self.device)
        rew = torch.from_numpy(self.rews[idxs]).to(device=self.device)
        next_obs = torch.from_numpy(self.obses[(idxs + 1) % self.buffer_size]).to(device=self.device)
        discount = torch.from_numpy(self.discounts[idxs]).to(device=self.device)
        return ReplayBufferSamples(obs, act, rew, next_obs, discount)
    
class ReplayBufferSamples(NamedTuple):
    obs: torch.Tensor
    act: torch.Tensor
    rew: torch.Tensor
    next_obs: torch.Tensor
    discount: torch.Tensor