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

    def step(self, action) -> dm_env.TimeStep:
        return self._convert_timestep(self.env.step(action))

    def reset(self) -> dm_env.TimeStep:
        return self._convert_timestep(self.env.reset())

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
    def __init__(self, env: dm_env.Environment, render_kwargs: dict | None = None):
        super().__init__(env)
        self.env = dm_control.suite.wrappers.pixels.Wrapper(self.env, render_kwargs=render_kwargs)
        self._observation_spec = self.env.observation_spec()['pixels']

    def _convert_timestep(self, timestep: dm_env.TimeStep):
        return timestep._replace(observation=timestep.observation['pixels'])

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