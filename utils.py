import dm_env
from rich.progress import track

def evaluate(policy, env: dm_env.Environment, episodes, progress_bar=False):
    total_rew = 0
    for _ in track(range(episodes), description="Evaluating...") if progress_bar else range(episodes):
        timestep = env.reset()
        while not timestep.last():
            act = policy(timestep.observation)
            timestep = env.step(act)
            total_rew += timestep.reward
    return total_rew / episodes