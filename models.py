import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class actor(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        self.fc1 = nn.Linear(self.observation_space, 512)
        self.fc2 = nn.Linear(512, 512)
        self.m = nn.Linear(512, self.action_space)
        self.std = nn.Linear(512, self.action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.m(x))
        std = torch.clip(self.std(x), min=0.000001, max=1)
        return mean, std

    def sample(self, x):
        mean, std = self.forward(x)
        dis = Normal(mean, std)
        action = dis.rsample()
        log_probs = torch.sum(dis.log_prob(action), dim=-1)
        return torch.clamp(action, -1, 1), log_probs

class critic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        self.fc1 = nn.Linear(self.observation_space + self.action_space, 512)
        self.fc2 = nn.Linear(512, 512)
        self.Q = nn.Linear(512, 1)

    def forward(self, x, a):
        x = torch.concat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        Q = self.Q(x)
        return Q