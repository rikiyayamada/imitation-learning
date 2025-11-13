import sys
from pathlib import Path
import math

if sys.platform == 'linux':
    if 'DISPLAY' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'
        print("MUJOCO_GL=egl")

import hydra
import dm_control.suite
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parent.parent))
from dmc.wrappers import PixelObservation, ActionRepeat
from rl import DrQv2, ReplayBuffer

@hydra.main(config_path='../configs', config_name=Path(__file__).stem, version_base=None)
def main(cfg):
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    env = dm_control.suite.load(cfg.domain, cfg.task, cfg.task_kwargs, cfg.environment_kwargs)
    env = PixelObservation(env, cfg.render_kwargs, cfg.num_stack)
    env = ActionRepeat(env, cfg.num_action_repeat)
    cfg.obs_shape = env.observation_spec().shape
    cfg.action_dim = env.action_spec().shape[0]
    cfg.device = torch.accelerator.current_accelerator().type
    rl_agent: DrQv2 = hydra.utils.instantiate(cfg.rl_agent)
    replay_buffer: ReplayBuffer = hydra.utils.instantiate(cfg.replay_buffer)
    writer = SummaryWriter(output_dir)
    timestep = env.reset()
    return_ = 0
    for step_c in tqdm(range(cfg.total_steps), dynamic_ncols=True):
        obs = timestep.observation[-3:]
        if timestep.last():
            replay_buffer.add(obs, discount=timestep.discount)
            writer.add_scalar('return', return_, step_c)
            return_ = 0
            timestep = env.reset()
        else:
            if step_c < cfg.random_steps:
                action = np.random.uniform(-1.0, 1.0, cfg.action_dim)
            else:
                action = rl_agent.predict(timestep.observation, step_c)
            timestep = env.step(action)
            reward = timestep.reward
            return_ += reward
            replay_buffer.add(obs, action, reward)
            if not step_c < cfg.random_steps and step_c % cfg.update_rl_agent_every == 0:
                replay_data = replay_buffer.sample()
                log_info = rl_agent.update(*replay_data, step_c)
                for key, value in log_info.items():
                    writer.add_scalar(key, value, step_c)

OmegaConf.register_new_resolver('mul', lambda *args: math.prod(args))
OmegaConf.register_new_resolver('sum', lambda *args: sum(args))

if __name__ == '__main__':
    main()