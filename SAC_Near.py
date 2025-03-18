import torch
import torch.nn as nn
from models import actor, critic
from buffer.near_buffer import buffer
import numpy

class SAC:
    def __init__(self, 
                 observation_space, 
                 action_space, 
                 maxlen=50000, 
                 batch_size=32,
                 gamma=0.99, 
                 tau=0.005, 
                 lr=0.0001,
                 entropy=0.1,
                 device='cuda'):
        self.observation_space = observation_space
        self.action_space = action_space
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.entropy = entropy
        
        self.memory = buffer(self.maxlen, self.batch_size)
        self.actor = actor(observation_space, action_space).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), 
                                                lr=self.lr)

        self.critic_1 = critic(observation_space, action_space).to(device)
        self.critic_2 = critic(observation_space, action_space).to(device)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), 
                                                lr=self.lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), 
                                                lr=self.lr)

        self.critic_1_target = critic(observation_space, action_space).to(device)
        self.critic_2_target = critic(observation_space, action_space).to(device)

        self.hard_update_single(self.critic_1_target, self.critic_1)
        self.hard_update_single(self.critic_2_target, self.critic_2)
        
    def soft_update_single(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
             
    def hard_update_single(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def save(self, s, a, r, s_, t):
        self.memory.save(s, a, r, s_, t)

    def get_action(self, s):
        s = torch.tensor(s).view(-1, self.observation_space).to(torch.float32).to(self.device)
        a, log_probs = self.actor.sample(s)
        return a.detach().cpu().numpy()[0]

    def train(self):
        batch = self.memory.get_batch()
        if batch == "N":
            return
        S, A, R, S_, T = batch
        S = S.to(self.device)
        A = A.to(self.device)
        R = R.to(self.device)
        S_ = S_.to(self.device)
        T = T.to(self.device)
        # compute target critic function.
        A_, log_probs_ = self.actor.sample(S)
        log_probs_ = log_probs_.unsqueeze(1)
        Qt1 = self.critic_1_target(S_, A_) - self.entropy * log_probs_
        Qt2 = self.critic_2_target(S_, A_) - self.entropy * log_probs_
        Y = R + self.gamma*(1-T)*torch.min(Qt1, Qt2)
        Y = Y.detach()
        # update critic.
        Q1 = self.critic_1(S, A)
        loss_c1 = nn.MSELoss()(Q1, Y)
        self.critic_1_optimizer.zero_grad()
        loss_c1.backward()
        self.critic_1_optimizer.step()
        
        Q2 = self.critic_2(S, A)
        loss_c2 = nn.MSELoss()(Q2, Y)
        self.critic_2_optimizer.zero_grad()
        loss_c2.backward()
        self.critic_2_optimizer.step()
        # update policy.
        A_c, log_probs_c = self.actor.sample(S)
        log_probs_c = log_probs_c.unsqueeze(1)
        Q1 = self.critic_1(S, A_c)
        Q2 = self.critic_2(S, A_c)
        Q = torch.min(Q1, Q2)
        loss = - torch.mean(Q - self.entropy * log_probs_c)
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        # update target
        self.soft_update_single(self.critic_1_target, self.critic_1, self.tau)
        self.soft_update_single(self.critic_2_target, self.critic_2, self.tau)