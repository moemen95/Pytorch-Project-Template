"""
Replay Memory class
Adapted from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
date: 1st of June 2018
"""
import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, config):
        self.config = config

        self.capacity = self.config.memory_capacity
        self.memory = []
        self.position = 0

    def length(self):
        return len(self.memory)

    def push_transition(self, *args):
        if self.length() < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity  # for the cyclic buffer

    def sample_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch
