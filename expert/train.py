import sys
from pathlib import Path
import math

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import hydra
from omegaconf import OmegaConf
import dm_control.suite
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from wrappers import FlattenObservation
from tdpg import TDPG
from replay_buffer import ReplayBuffer

@hydra.main(config_path='config', config_name='train', version_base=None)
def main(cfg):
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    env = dm_control.suite.load(cfg.domain, cfg.task, cfg.task_kwargs, cfg.environment_kwargs)
    env = FlattenObservation(env)
    cfg.obs_dim = env.observation_spec().shape[0]
    cfg.action_dim = env.action_spec().shape[0]
    agent: TDPG = hydra.utils.instantiate(cfg.tdpg)
    obs_spec = env.observation_spec()
    action_spec = env.action_spec()
    replay_buffer: ReplayBuffer = hydra.utils.instantiate(cfg.replay_buffer, obs_spec=obs_spec, action_spec=action_spec)
    writer = SummaryWriter(output_dir)
    try:
        timestep = env.reset()
        return_ = 0
        for step_c in tqdm(range(cfg.total_steps), dynamic_ncols=True):
            if timestep.last():
                replay_buffer.add(timestep.observation)
                writer.add_scalar('return', return_, step_c)
                return_ = 0
                timestep = env.reset()
            else:
                obs = timestep.observation
                if step_c < cfg.random_steps:
                    action = np.random.uniform(action_spec.minimum, action_spec.maximum, action_spec.shape)
                else:
                    action = agent.predict(obs, step_c, deterministic=False)
                timestep = env.step(action)
                reward = timestep.reward
                return_ += reward
                discount = timestep.discount
                replay_buffer.add(obs, action, reward, discount)
                if not step_c < cfg.random_steps:
                    replay_data = replay_buffer.sample()
                    log_info = agent.update(*replay_data, step_c)
                    for key, value in log_info.items():
                        writer.add_scalar(key, value, step_c)
    except KeyboardInterrupt:
        print(f'Interrupted at {step_c} steps')
    torch.save(agent.actor.state_dict(), output_dir/'state_dict.pt')

OmegaConf.register_new_resolver('mul', lambda *args: math.prod(args))
OmegaConf.register_new_resolver('sum', lambda *args: sum(args))

if __name__ == '__main__':
    main()