import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from buffers.near_buffer import buffer
from models import actor, critic

class DDPG:
    def __init__(self, 
                 observation_space, 
                 action_space, 
                 batch_size = 128, 
                 maxlen=50000, 
                 gamma=0.99, 
                 tau=0.005, 
                 lr=0.0001,
                 device='cuda'):
        self.observation_space = observation_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.device = device

        self.buffer = buffer(maxlen=self.maxlen, batch_size=self.batch_size)

        self.actor = actor(self.observation_space, self.action_space)
        self.actor_target = actor(self.observation_space, self.action_space)

        self.critic = critic(self.observation_space, self.action_space)
        self.critic_target = critic(self.observation_space, self.action_space)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.actor.to(self.device)
        self.critic.to(self.device)
        self.actor_target.to(self.device)
        self.critic_target.to(self.device)
                
        self.hard_update()

    def soft_update(self):
        self.soft_update_single(self.actor_target, self.actor, self.tau)
        self.soft_update_single(self.critic_target, self.critic, self.tau)

    def hard_update(self):
        self.hard_update_single(self.actor_target, self.actor)
        self.hard_update_single(self.critic_target, self.critic)

    def soft_update_single(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
             
    def hard_update_single(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def get_action(self, observation, epsilon=0.1):
        e = random.random()        
        observation = torch.tensor(observation).view(1, -1).to(self.device).to(torch.float32)
        action = self.actor(observation)[0]
        if e < epsilon:
            action = torch.clamp(action + torch.rand(action.shape).to(self.device)/3, min=-1, max=1)
            return action.detach().cpu().numpy()
        return action.detach().cpu().numpy()

    def save(self, s, a, r, s_, t):
        self.buffer.save(s, a, r, s_, t)

    def train(self):
        batch = self.buffer.get_batch()
        if batch == "N":
            return 
        S, A, R, S_, T = batch
        S = S.to(self.device)
        A = A.to(self.device)
        R = R.to(self.device)
        S_ = S_.to(self.device)
        T = T.to(self.device)

        #train critic.
        A_ = self.actor_target(S_)
        next_q = self.critic_target(S_, A_)
        q_values = R + self.gamma * next_q * (1 - T)

        self.critic_optimizer.zero_grad()
        current_q = self.critic(S, A)
        critic_loss = nn.MSELoss()(q_values, current_q)
        #critic_loss = torch.clamp(critic_loss, min=-.1, max=.1)
        critic_loss.backward()
        self.critic_optimizer.step()

        #train actor.
        self.actor_optimizer.zero_grad()

        actor_loss = -self.critic(S, self.actor(S))
        actor_loss = torch.mean(actor_loss)
        #actor_loss = torch.clamp(actor_loss, min=-.1, max=.1)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update()