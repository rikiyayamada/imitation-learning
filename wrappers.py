import dm_env
import dm_env.specs
import dm_control.suite
import dm_control.suite.wrappers.pixels
import numpy as np

class Wrapper(dm_env.Environment):
    def __init__(self, environment: dm_env.Environment):
        self.environment = environment
        self._action_spec = self.environment.action_spec()
        self._observation_spec = self.environment.observation_spec()
    
    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError(name)
        return getattr(self.environment, name)
    
    def _convert_timestep(self, timestep: dm_env.TimeStep):
        return timestep

    def step(self, action) -> dm_env.TimeStep:
        return self._convert_timestep(self.environment.step(action))

    def reset(self) -> dm_env.TimeStep:
        return self._convert_timestep(self.environment.reset())

    def action_spec(self):
        return self._action_spec

    def discount_spec(self):
        return self.environment.discount_spec()

    def observation_spec(self):
        return self._observation_spec

    def reward_spec(self):
        return self.environment.reward_spec()

    def close(self):
        return self.environment.close()

class FlattenObservation(Wrapper):
    def __init__(self, environment: dm_env.Environment):
       super().__init__(environment) 
       dim = sum(np.prod(spec.shape) for spec in self.environment.observation_spec().values())
       self._observation_spec = dm_env.specs.Array(shape=(dim,), dtype=np.float32)
    
    def _convert_timestep(self, timestep: dm_env.TimeStep):
        flatten_observation = np.concatenate([obs.ravel() for obs in timestep.observation.values()], axis=0).astype(np.float32)
        return timestep._replace(observation=flatten_observation)

class ReacherTerminate(Wrapper):
    def __init__(self, environment: dm_env.Environment):
        super().__init__(environment)
    
    def _convert_timestep(self, timestep: dm_env.TimeStep):
        if self.environment.physics.finger_to_target_dist() <= self.environment.physics.named.model.geom_size[['target', 'finger'], 0].sum():
            return timestep._replace(step_type=dm_env.StepType.LAST, reward=1.0, discount=0.0)
        else:
            return timestep._replace(reward=-0.1)

original_load = dm_control.suite.load
def patched_load(domain_name, task_name, task_kwargs=None, environment_kwargs=None, visualize_reward=False):
    if domain_name == 'reacher_terminate':
        base_domain_name = 'reacher'
        wrapper = ReacherTerminate
    else:
        base_domain_name = domain_name
        wrapper = lambda env: env
    env = original_load(base_domain_name, task_name, task_kwargs, environment_kwargs, visualize_reward)
    env = wrapper(env)
    return env
dm_control.suite.load = patched_load