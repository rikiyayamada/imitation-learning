import queue
import math

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import IterableDataset, DataLoader

class Episode:
    __slots__ = 'obs', 'action', 'reward', 'discount', 'len'
    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.discount = None
        self.len = None

class RingBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.sum_tree_size = int(2 ** math.ceil(math.log2(self.buffer_size)))
        self.sum_tree = [0 for _ in range(2 * self.sum_tree_size)]
        self.episodes = [None for _ in range(self.buffer_size)]
        self.rng = None
        self.head = 0
        self.tail = 0
        self.num_episodes = 0
    
    @property
    def current_buffer_size(self):
        return self.sum_tree[1]

    def popleft(self):
        if self.num_episodes == 0:
            raise IndexError()
        self.episodes[self.tail] = None
        self.sum_tree_set(self.tail, 0)
        self.tail = (self.tail + 1) % self.buffer_size
        self.num_episodes -= 1
    
    def sum_tree_set(self, idx, val):
        idx += self.sum_tree_size
        self.sum_tree[idx] = val
        idx //= 2
        while idx >= 1:
            l_child_idx = 2 * idx
            r_child_idx = 2 * idx + 1
            self.sum_tree[idx] = self.sum_tree[l_child_idx] + self.sum_tree[r_child_idx]
            idx //= 2

    def append(self, episode: Episode):
        while self.current_buffer_size + episode.len > self.buffer_size:
            self.popleft()
        self.episodes[self.head] = episode
        self.sum_tree_set(self.head, episode.len)
        self.head = (self.head + 1) % self.buffer_size
        self.num_episodes += 1
    
    def sample(self) -> tuple[Episode, int]:
        i = self.rng.integers(0, self.current_buffer_size)
        idx = 1
        while idx < self.sum_tree_size:
            l_child_idx = 2 * idx
            r_child_idx = 2 * idx + 1
            if i >= self.sum_tree[l_child_idx]:
                idx = r_child_idx
                i -= self.sum_tree[l_child_idx]
            else:
                idx = l_child_idx
        j = idx - self.sum_tree_size
        return self.episodes[j], i
    
class Worker(IterableDataset):
    def __init__(
        self,
        queues,
        buffer_size,
        gamma,
        obs_shape,
        n_step,
        num_stack,
        pixel_obs,
    ):
        self.queues = queues
        self.gamma = gamma
        self.n_step = n_step
        self.num_stack = num_stack
        self.pixel_obs = pixel_obs
        self.obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1]) if self.pixel_obs else obs_shape
        self.buffer = RingBuffer(buffer_size)
    
    def stack(self, episode: Episode, i):
        frames = np.empty((self.num_stack, *self.obs_shape), dtype=np.uint8)
        frames[-1] = episode.obs[i]
        s = self.num_stack - 2
        for k in range(i - 1, i - self.num_stack, -1):
            if k < 0:
                frames[:s + 1] = frames[s + 1][None, ...]
                break
            frames[s] = episode.obs[k]
            s -= 1
        obs = frames.reshape((self.num_stack * self.obs_shape[0], *self.obs_shape[1:]))
        return obs
    
    def sample(self):
        episode, i = self.buffer.sample()
        if self.pixel_obs:
            obs = self.stack(episode, i)
        else:
            obs = episode.obs[i]
        action = episode.action[i]
        reward = 0.0
        discount = 1.0
        for k in range(i, i + self.n_step):
            reward += episode.reward[k] * discount
            discount *= self.gamma
            if k + 1 == episode.len:
                discount *= episode.discount
                break
        if self.pixel_obs:
            next_obs = self.stack(episode, k + 1)
        else:
            next_obs = episode.obs[k + 1]
        return obs, action, reward, next_obs, discount
    
    def init(self):
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        self.queue: mp.Queue = self.queues[worker_id % len(self.queues)]
        self.buffer.rng = np.random.default_rng(torch.initial_seed())
    
    def fetch(self):
        try:
            while True:
                episode = self.queue.get_nowait()
                self.buffer.append(episode)
        except queue.Empty:
            pass

    def __iter__(self):
        self.init()
        episode = self.queue.get()
        self.buffer.append(episode)
        while True:
            self.fetch()
            yield self.sample()

class ReplayBuffer:
    Worker = Worker
    Episode = Episode

    def __init__(
        self,
        batch_size,
        gamma,
        obs_shape,
        device,
        buffer_size=1_000_000,
        num_workers=1,
        n_step=1,
        num_stack=1,
        pixel_obs=False,
    ):
        self.batch_size = batch_size
        self.gamma = gamma
        self.obs_shape = obs_shape
        self.device = torch.device(device)
        self.num_workers = num_workers
        self.n_step = n_step
        self.num_stack = num_stack
        self.pixel_obs = pixel_obs

        self.queues = [mp.Queue(maxsize=10) for _ in range(num_workers)]
        worker = self.Worker(self.queues, buffer_size // num_workers, self.gamma, self.obs_shape, self.n_step, self.num_stack, self.pixel_obs)
        data_loader = DataLoader(
            dataset=worker,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.data_loader_iter = iter(data_loader)

        self.episode = self.Episode()
        self.num_episodes = 0
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray | None = None,
        reward: np.ndarray | None = None,
        discount: float | None = None,
    ):
        self.episode.obs.append(obs)
        if action is not None:
            self.episode.action.append(action)
            self.episode.reward.append(reward)
        elif discount is not None:
            if len(self.episode.action) > 0:
                self.episode.obs = np.array(self.episode.obs, dtype=np.uint8 if self.pixel_obs else np.float32)
                self.episode.action = np.array(self.episode.action, dtype=np.float32)
                self.episode.reward = np.array(self.episode.reward, dtype=np.float32)
                self.episode.discount = discount
                self.episode.len = len(self.episode.action)
                self.queues[self.num_episodes % self.num_workers].put(self.episode)
                self.num_episodes += 1
            self.episode = self.Episode()
        else:
            raise ValueError()
    
    def sample(self):
        data = next(self.data_loader_iter)
        return tuple(tensor.to(device=self.device, dtype=torch.float32) for tensor in data)