from collections import deque
import numpy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class buffer:
    def __init__(self, maxlen=50000, batch_size=32):
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.maxlen)
        self.last = deque(maxlen=self.batch_size//2)
    
    def save(self, s, a, r, s_, t):
        self.buffer.append([s, a, r, s_, t])
        self.last.append([s, a, r, s_, t])
    
    def get_batch(self):
        if len(self.buffer) < self.batch_size:
            return "N"
        batch = random.sample(self.buffer, k=self.batch_size//2) + list(self.last)[-self.batch_size//2:]
        S = torch.tensor(numpy.array([val[0] for val in batch])).view(self.batch_size, -1).to(torch.float32)
        A = torch.tensor(numpy.array([val[1] for val in batch])).view(self.batch_size, -1).to(torch.float32)
        R = torch.tensor(numpy.array([val[2] for val in batch])).view(self.batch_size, -1).to(torch.float32)
        S_ = torch.tensor(numpy.array([val[3] for val in batch])).view(self.batch_size, -1).to(torch.float32)
        T = torch.tensor(numpy.array([val[4] for val in batch])).view(self.batch_size, -1).to(torch.float32)
        return S, A, R, S_, T
