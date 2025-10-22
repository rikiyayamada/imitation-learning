import sys
from pathlib import Path

import dm_env
import dm_control.suite
from rich.progress import track
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue
import torch
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from wrappers import FlattenObservation
from networks import Actor

@hydra.main(config_path='config', config_name='collect.yaml', version_base=None)
def main(cfg):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() and cfg.device != 'cpu' else 'cpu'
    state_dict_path = Path(cfg.state_dict_path)
    train_cfg = OmegaConf.load(state_dict_path.parent/'.hydra/config.yaml')
    with dm_control.suite.load(train_cfg.domain, train_cfg.task) as env:
        env = FlattenObservation(env)
        actor = Actor(train_cfg.actor, env.action_spec().shape[0]).to(device)
        dummy_obs = np.zeros(env.observation_spec().shape)
        with torch.no_grad():
            actor.predict(dummy_obs)
        actor.load_state_dict(torch.load(state_dict_path, map_location=device))
        output_dir = Path(f'{HydraConfig.get().runtime.output_dir}')
        collector = Collector(env, actor.predict, cfg.render_kwargs, output_dir)
        if cfg.timesteps is not None:
            for _ in track(range(cfg.timesteps), description="Collecting..."):
                collector()
            collector.save(f'{cfg.timesteps}timesteps')
        elif cfg.episodes is not None:
            for _ in track(range(cfg.episodes), description="Collecting..."):
                while not collector.timestep.last():
                    collector()
                collector()
            collector.save(f'{cfg.episodes}episodes')
        else:
            raise MissingMandatoryValue('either timesteps or episodes required')

OmegaConf.register_new_resolver("task_name", lambda path: Path(path).parent.parent.name)

class Collector:
    def __init__(self, env: dm_env.Environment, policy, render_kwargs: dict, output_dir: Path):
        self.env = env
        self.policy = policy
        self.render_kwargs = render_kwargs
        self.output_dir = output_dir
        self.timestep = self.env.reset()
        self.frames = []
        self.acts = []
        self.episode_end_idxs = []
        self.timesteps = 0
    
    def __call__(self):
        self.frames.append(self.env.physics.render(**self.render_kwargs))
        if self.timestep.last():
            act = np.zeros(self.env.action_spec().shape)
            self.episode_end_idxs.append(self.timesteps)
            self.timestep = self.env.reset()
        else:
            act = self.policy(self.timestep.observation)
            self.timestep = self.env.step(act)
        self.acts.append(act)
        self.timesteps += 1
    
    def save(self, file_name):
        np.savez(self.output_dir/file_name, obs=np.array(self.frames, dtype=np.uint8), act=np.array(self.acts, dtype=np.float32), episode_end_idx=np.array(self.episode_end_idxs, dtype=np.int32))

if __name__ == '__main__':
    main()