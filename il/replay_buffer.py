from typing import override
import queue

import numpy as np

import rl.replay_buffer

class Episode:
    __slots__ = 'obs', 'action', 'len', 'is_absorbing'
    def __init__(self):
        self.obs = []
        self.action = []
        self.is_absorbing = None
        self.len = None

class Worker(rl.replay_buffer.Worker):
    def n_step_obs_action(self, episode, i):
        if self.pixel_obs:
            obs = np.empty((self.n_step, self.num_stack * self.obs_shape[0], *self.obs_shape[1:]), dtype=np.uint8)
            for j in range(self.n_step):
                obs[j] = self.stack(episode, i + j)
        else:
            obs = episode.obs[i:i + self.n_step]
        action = episode.action[i:i + self.n_step]
        is_absorbing = episode.is_absorbing[i:i + self.n_step]
        return obs, action, is_absorbing
        
    @override
    def sample(self):
        episode, i = self.buffer.sample()
        obs, action, is_absorbing = self.n_step_obs_action(episode, i)
        if self.pixel_obs:
            next_obs = self.stack(episode, i + self.n_step)
        else:
            next_obs = episode.obs[i + self.n_step]
        next_is_absorbing = np.array([episode.is_absorbing[i + self.n_step]])
        discount = self.gamma**self.n_step
        return obs, action, is_absorbing, next_obs, next_is_absorbing, discount
    
class ReplayBuffer(rl.ReplayBuffer):
    Worker = Worker
    Episode = Episode

    @override
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray | None = None,
        discount: float | None = None,
    ):
        self.episode.obs.append(obs)
        if action is not None:
            self.episode.action.append(action)
        elif discount is not None:
            if len(self.episode.action) > 0:
                self.episode.len = len(self.episode.action)
                if discount == 0.0:
                    self.episode.obs.extend([obs] * (self.n_step + 1))
                    self.episode.action.extend([self.episode.action[-1]] * self.n_step)
                    self.episode.len += 1
                elif discount == 1.0:
                    self.episode.obs.append(obs)
                    self.episode.len -= (self.n_step - 1)
                else:
                    raise ValueError()
                self.episode.obs = np.array(self.episode.obs, dtype=np.uint8 if self.pixel_obs else np.float32)
                self.episode.action = np.array(self.episode.action, dtype=np.float32)
                self.episode.is_absorbing = np.zeros(len(self.episode.obs), dtype=bool)
                self.episode.is_absorbing[self.episode.len - 1:] = True
                self.queues[self.num_episodes % self.num_workers].put(self.episode)
                self.num_episodes += 1
            self.episode = self.Episode()
        else:
            raise ValueError()

class ExpertWorker(Worker):
    @override
    def sample(self):
        episode, i = self.buffer.sample()
        return self.n_step_obs_action(episode, i)
    
    @override
    def __iter__(self):
        self.init()
        while True:
            episode = self.queue.get()
            if episode is None:
                break
            self.buffer.append(episode)
        while True:
            yield self.sample()

class ExpertReplayBuffer(ReplayBuffer):
    Worker = ExpertWorker

    @override
    def __init__(
        self,
        demo_path,
        batch_size,
        gamma,
        obs_shape,
        n_step,
        num_stack,
        device,
        pixel_obs,
        num_workers=1,
    ):
        demo = np.load(demo_path)
        buffer_size = len(demo['done'])
        super().__init__(
            batch_size,
            gamma,
            obs_shape,
            device,
            buffer_size,
            num_workers,
            n_step,
            num_stack,
            pixel_obs,
        )
        if self.pixel_obs:
            obs = demo['pixel_obs'].transpose(0, 3, 1, 2)
        else:
            obs = demo['obs']
        for i in range(len(demo['done'])):
            if not demo['done'][i]:
                self.add(obs[i], demo['action'][i])
            else:
                self.add(obs[i], discount=demo['discount'][i])
        for i in range(self.num_workers):
            self.queues[i].put(None)