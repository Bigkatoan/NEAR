import os
import torch
import gymnasium as gym
from SAC_Near import SAC
from torch.utils.tensorboard import SummaryWriter

def train(env_name, agent_type='SAC_Normal', episodes=10000, T=1000):
    path = f'models/{env_name}_{agent_type}'
    os.makedirs(path, exist_ok = True) 
    print(f'start training {agent_type} on {env_name}')
    env = env = gym.make(env_name)
    

    if agent_type == 'SAC_Normal':
        from SAC_Normal import SAC
        agent = SAC(env.observation_space.shape[0], env.action_space.shape[0])
    if agent_type == 'SAC_Near':
        from SAC_Near import SAC
        agent = SAC(env.observation_space.shape[0], env.action_space.shape[0])
    if agent_type == 'DDPG_Normal':
        from DDPG_Normal import DDPG
        agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0])
    if agent_type == 'DDPG_Near':
        from DDPG_Near import DDPG
        agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0])

    for episode in range(episodes):
        observation, _ = env.reset()
        if episode % 1000 == 0:
            torch.save(agent.actor, path + f'actor_{episode}.pt')
        score = 0
        for tm in range(T):
            s = observation
            a = agent.get_action(s)
            observation, r, t, _, _ = env.step(a)
            s_ = observation
            score += r
            agent.save(s, a, r, s_, t)
            agent.train()
            if t:
                break
        writer.add_scalar(f'{env_name}/{agent_type}/score', score, episode)
        torch.save(agent.actor, path + 'actor_last.pt')
    print('train complete')

if __name__ == "__main__":
    os.system('rm -rf runs')
    writer = SummaryWriter()
    envs = ['Ant-v5', 'HalfCheetah-v5', 'Hopper-v5', 'HumanoidStandup-v5', 'Humanoid-v5', 'InvertedDoublePendulum-v5', 'InvertedPendulum-v5', 'Pusher-v5', 'Reacher-v5', 'Swimmer-v5', 'Walker2d-v5']
    agents = ['SAC_Normal', 'SAC_Near', 'DDPG_Normal', 'DDPG_Near']
    for env in envs:
        for agent_type in agents:
            train(env, agent_type, episodes=10000, T=1000)