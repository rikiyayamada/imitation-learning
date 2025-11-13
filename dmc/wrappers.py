from collections import deque

import numpy as np
import dm_env
import dm_env.specs
import dm_control.suite
import dm_control.suite.wrappers.pixels

class Wrapper(dm_env.Environment):
    def __init__(self, env: dm_env.Environment):
        self.env = env
        self._action_spec = self.env.action_spec()
        self._observation_spec = self.env.observation_spec()
    
    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError(name)
        return getattr(self.env, name)
    
    def _convert_timestep(self, timestep: dm_env.TimeStep):
        return timestep

    def reset(self) -> dm_env.TimeStep:
        return self._convert_timestep(self.env.reset())

    def step(self, action) -> dm_env.TimeStep:
        return self._convert_timestep(self.env.step(action))

    def action_spec(self):
        return self._action_spec

    def discount_spec(self):
        return self.env.discount_spec()

    def observation_spec(self):
        return self._observation_spec

    def reward_spec(self):
        return self.env.reward_spec()

    def close(self):
        return self.env.close()

class FlattenObservation(Wrapper):
    def __init__(self, env: dm_env.Environment):
        super().__init__(env) 
        dim = sum(np.prod(spec.shape) for spec in self.env.observation_spec().values())
        self._observation_spec = dm_env.specs.Array(shape=(dim,), dtype=np.float32)
    
    def _convert_timestep(self, timestep: dm_env.TimeStep):
        flatten_observation = np.concatenate([obs.ravel() for obs in timestep.observation.values()], axis=0).astype(np.float32)
        return timestep._replace(observation=flatten_observation)

class PixelObservation(Wrapper):
    def __init__(self, env: dm_env.Environment, render_kwargs: dict | None = None, num_stack: int = 1):
        super().__init__(env)
        self.env = dm_control.suite.wrappers.pixels.Wrapper(self.env, render_kwargs=render_kwargs)
        self._observation_spec = self.env.observation_spec()['pixels']
        self.num_stack = num_stack
        self.frames = deque([], maxlen=self.num_stack)

    def _convert_timestep(self, timestep: dm_env.TimeStep):
        return timestep._replace(observation=timestep.observation['pixels'])
    
    def transpose_and_reshape(self):
        transposed_obs = np.stack(self.frames).transpose(0, 3, 1, 2)
        reshaped_obs = transposed_obs.reshape(-1, transposed_obs.shape[2], transposed_obs.shape[3])
        return reshaped_obs
    
    def reset(self):
        timestep = super().reset()
        self.frames.extend([timestep.observation] * self.num_stack)
        return timestep._replace(observation=self.transpose_and_reshape())

    def step(self, action):
        timestep = super().step(action)
        self.frames.append(timestep.observation)
        return timestep._replace(observation=self.transpose_and_reshape())

class ActionRepeat(Wrapper):
    def __init__(self, env: dm_env.Environment, num_action_repeats: int):
        super().__init__(env)
        self.num_action_repeats = num_action_repeats
    
    def step(self, action):
        reward = 0.0
        for _ in range(self.num_action_repeats):
            timestep = super().step(action)
            reward += timestep.reward
            if timestep.last():
                break
        return timestep._replace(reward=reward)

class ReacherTerminate(Wrapper):
    def __init__(self, env: dm_env.Environment):
        super().__init__(env)
    
    def _convert_timestep(self, timestep: dm_env.TimeStep):
        if self.env.physics.finger_to_target_dist() <= self.env.physics.named.model.geom_size[['target', 'finger'], 0].sum():
            return timestep._replace(step_type=dm_env.StepType.LAST, reward=1.0, discount=0.0)
        else:
            return timestep._replace(reward=-0.1)

original_load = dm_control.suite.load
def patched_load(domain_name, task_name, task_kwargs=None, env=None, visualize_reward=False):
    if domain_name == 'reacher_terminate':
        base_domain_name = 'reacher'
        wrapper = ReacherTerminate
    else:
        base_domain_name = domain_name
        wrapper = lambda env: env
    env = original_load(base_domain_name, task_name, task_kwargs, env, visualize_reward)
    env = wrapper(env)
    return env
dm_control.suite.load = patched_load