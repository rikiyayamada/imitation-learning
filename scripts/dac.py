import sys
import os
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
from dmc.wrappers import PixelObservation, ActionRepeat, FlattenObservation
from il import DAC

@hydra.main(config_path='../configs', config_name=Path(__file__).stem, version_base=None)
def main(cfg):
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    env = dm_control.suite.load(cfg.domain, cfg.task, cfg.task_kwargs, cfg.environment_kwargs)
    env = PixelObservation(env, cfg.render_kwargs, cfg.num_stack) if cfg.pixel_obs else FlattenObservation(env)
    env = ActionRepeat(env, cfg.num_action_repeat)
    cfg.obs_shape = env.observation_spec().shape
    cfg.action_dim = env.action_spec().shape[0]
    if not cfg.pixel_obs:
        cfg.h_dim = env.observation_spec().shape[0]
        cfg.trunk = None
        cfg.encoder = None
        cfg.aug = None
    cfg.device = torch.accelerator.current_accelerator().type if cfg.pixel_obs else 'cpu'
    il_agent: DAC = hydra.utils.instantiate(cfg.il_agent)
    writer = SummaryWriter(output_dir)
    timestep = env.reset()
    return_ = 0
    tqdm_kwargs = {'iterable': range(cfg.total_steps), 'dynamic_ncols': True}
    try:
        job_num = hydra.core.hydra_config.HydraConfig.get().job.num
        tqdm_kwargs |= {'position': job_num, 'desc': f"#{job_num}"}
    except Exception:
        pass
    for step_c in tqdm(**tqdm_kwargs):
        obs = timestep.observation[-3:] if cfg.pixel_obs else timestep.observation
        if timestep.last():
            il_agent.replay_buffer.add(obs, discount=timestep.discount)
            writer.add_scalar('return', return_, step_c)
            return_ = 0
            timestep = env.reset()
        else:
            if step_c < cfg.random_steps:
                action = np.random.uniform(-1.0, 1.0, cfg.action_dim)
            else:
                action = il_agent.predict(timestep.observation, step_c)
            timestep = env.step(action)
            return_ += timestep.reward
            il_agent.replay_buffer.add(obs, action)
            if not step_c < cfg.random_steps:
                log_info = il_agent.update(step_c)
                for key, value in log_info.items():
                    writer.add_scalar(key, value, step_c)

OmegaConf.register_new_resolver('mul', lambda *args: math.prod(args))
OmegaConf.register_new_resolver('sum', lambda *args: sum(args))

if __name__ == '__main__':
    main()