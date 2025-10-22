import sys
from pathlib import Path

import hydra
import dm_control.suite
import torch

sys.path.append(str(Path(__file__).parent.parent))
from networks import Actor, Critic
from sac import SAC
from replay_buffer import ReplayBuffer
from wrappers import FlattenObservation

@hydra.main(config_path='config', config_name='train.yaml', version_base=None)
def main(cfg):
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    with dm_control.suite.load(cfg.domain, cfg.task) as env:
        env = FlattenObservation(env)
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() and cfg.device != 'cpu' else 'cpu'
        print(f'device: {device}')
        actor = Actor(cfg.actor, act_dim=env.action_spec().shape[0])
        critic = Critic(cfg.critic)
        critic_target = Critic(cfg.critic)
        dummy_obs = torch.zeros(1, *env.observation_spec().shape)
        dummy_act = torch.zeros(1, *env.action_spec().shape)
        with torch.no_grad():
            critic(dummy_obs, dummy_act)
            critic_target(dummy_obs, dummy_act)
        critic_target.load_state_dict(critic.state_dict())
        replay_buffer = ReplayBuffer(env.observation_spec().shape, env.action_spec().shape, device=device, **cfg.replay_buffer)
        sac = SAC(env=env, actor=actor, critic=critic, critic_target=critic_target, replay_buffer=replay_buffer, log_dir=output_dir, device=device, **cfg.sac)
        try:
            sac.learn(cfg.total_timesteps)
        except KeyboardInterrupt:
            print('Learning interrupted')
            sac.eval()
        torch.save(sac.best_state_dict, output_dir/f'best_state_dict_{sac.best_score_timesteps}.pt')
        torch.save(sac.actor.state_dict(), output_dir/f'final_state_dict_{sac.timesteps}.pt')

if __name__ == '__main__':
    main()