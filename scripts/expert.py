import sys
from pathlib import Path
import math

import hydra
import dm_control.suite
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append(str(Path(__file__).parent.parent))
from dmc.wrappers import FlattenObservation, ActionRepeat
from rl import DrQv2, ReplayBuffer

@hydra.main(config_path='../configs', config_name=Path(__file__).stem, version_base=None)
def main(cfg):
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    env = dm_control.suite.load(cfg.domain, cfg.task, cfg.task_kwargs, cfg.environment_kwargs)
    env = FlattenObservation(env)
    env = ActionRepeat(env, cfg.num_action_repeat)
    cfg.obs_shape = env.observation_spec().shape
    cfg.obs_dim = env.observation_spec().shape[0]
    cfg.action_dim = env.action_spec().shape[0]
    rl_agent: DrQv2 = hydra.utils.instantiate(cfg.rl_agent)
    replay_buffer: ReplayBuffer = hydra.utils.instantiate(cfg.replay_buffer)
    writer = SummaryWriter(output_dir)
    try:
        timestep = env.reset()
        return_ = 0
        for step_c in tqdm(range(cfg.total_steps), dynamic_ncols=True):
            obs = timestep.observation
            if timestep.last():
                replay_buffer.add(obs, discount=timestep.discount)
                writer.add_scalar('return', return_, step_c)
                return_ = 0
                timestep = env.reset()
            else:
                if step_c < cfg.random_steps:
                    action = np.random.uniform(-1.0, 1.0, cfg.action_dim)
                else:
                    action = rl_agent.predict(obs, step_c)
                timestep = env.step(action)
                reward = timestep.reward
                return_ += reward
                replay_buffer.add(obs, action, reward)
                if not step_c < cfg.random_steps:
                    replay_data = replay_buffer.sample()
                    log_info = rl_agent.update(*replay_data, step_c)
                    for key, value in log_info.items():
                        writer.add_scalar(key, value, step_c)
    except KeyboardInterrupt:
        print(f'Interrupted at {step_c} steps')
    torch.save(rl_agent.actor.state_dict(), output_dir/'state_dict.pt')

@hydra.main(config_path='../configs', config_name='demo', version_base=None)
def demo(cfg):
    if cfg.total_episodes is None and cfg.total_steps is None:
        raise ValueError('Either total_episodes or total_steps required')
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    train_cfg = OmegaConf.load(Path(cfg.state_dict_path).parent/'.hydra/config.yaml')
    env = dm_control.suite.load(train_cfg.domain, train_cfg.task, train_cfg.task_kwargs, train_cfg.environment_kwargs)
    env = FlattenObservation(env)
    env = ActionRepeat(env, train_cfg.num_action_repeat)
    train_cfg.obs_dim = env.observation_spec().shape[0]
    train_cfg.action_dim = env.action_spec().shape[0]
    rl_agent: DrQv2 = hydra.utils.instantiate(train_cfg.rl_agent)
    rl_agent.actor.load_state_dict(torch.load(Path(cfg.state_dict_path)))
    obs_list = []
    pixel_obs_list = []
    action_list = []
    discount_list = []
    done_list = []
    step_c = 0
    episode_c = 0
    file_name = None
    timestep = env.reset()
    while True:
        while timestep.last():
            timestep = env.reset()
        obs_list.append(timestep.observation)
        pixel_obs_list.append(env.physics.render(**cfg.render_kwargs))
        action = rl_agent.predict(timestep.observation)
        action_list.append(action)
        discount_list.append(1.0)
        done_list.append(False)
        timestep = env.step(action)
        step_c += 1
        if timestep.last():
            obs_list.append(timestep.observation)
            pixel_obs_list.append(env.physics.render(**cfg.render_kwargs))
            action_list.append(np.zeros_like(action))
            discount_list.append(timestep.discount)
            done_list.append(True)
            episode_c += 1
            if cfg.total_episodes is not None and episode_c == cfg.total_episodes:
                file_name = f'{episode_c}episodes'
            elif cfg.total_steps is not None and step_c >= cfg.total_steps:
                file_name = f'{step_c}steps'
            if file_name is not None:
                np.savez_compressed(
                    output_dir/file_name,
                    obs=np.array(obs_list, dtype=np.float32),
                    pixel_obs=np.array(pixel_obs_list, dtype=np.uint8),
                    action=np.array(action_list, dtype=np.float32),
                    discount=np.array(discount_list, dtype=np.float32),
                    done=np.array(done_list, dtype=bool),
                )
                break

OmegaConf.register_new_resolver('mul', lambda *args: math.prod(args))
OmegaConf.register_new_resolver('sum', lambda *args: sum(args))
OmegaConf.register_new_resolver('domain_task', lambda path: f'{Path(path).parts[2]}')

def animate(demo_path):
    frames = np.load(demo_path)['pixel_obs']
    print(f'total steps: {len(frames)}')
    fig, ax = plt.subplots()
    ax.axis('off')
    im = ax.imshow(frames[0])
    def update(frame):
        im.set_data(frames[frame])
        return [im]
    interval = 33
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=True)
    state = {'running': True, 'interval': interval}
    def restart_if_running():
        if state['running']:
            ani.event_source.stop()
            ani.event_source.start()
    def on_key(event):
        if event.key == '=':
            state['interval'] = max(1, state['interval'] - 10)
            ani.event_source.interval = state['interval'] 
            restart_if_running()
        elif event.key == '-':
            state['interval'] += 10
            ani.event_source.interval = state['interval']
            restart_if_running()
        elif event.key == ' ':
            if state['running']:
                ani.event_source.stop()
                state['running'] = False
            else:
                ani.event_source.interval = state['interval'] 
                ani.event_source.start()
                state['running'] = True
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

if __name__ == '__main__':
    if sys.argv[1].startswith('state_dict_path='):
        demo()
    elif sys.argv[1].startswith('demo_path='):
        animate(sys.argv[1].split('=', 1)[1])
    else:
        main()