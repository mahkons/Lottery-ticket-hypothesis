import torch
import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.position = 0
    
    def push(self, state, action, next_state, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(
            state.detach(),
            action.detach(),
            next_state.detach(),
            reward.detach(),
            done.detach(),
        )
        
        self.position += 1
        if self.position == self.capacity:
            self.position = 0
            
    def sample(self, batch_size):
        return self.get_transitions(self.sample_positions(batch_size))

    def get_transitions(self, positions):
        transitions = [self.memory[pos] for pos in positions]

        batch = Transition(*zip(*transitions))
        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        reward = torch.cat(batch.reward)
        next_state = torch.cat(batch.next_state)
        done = torch.cat(batch.done)

        return state, action, next_state, reward, done

    def sample_positions(self, batch_size):
        return random.sample(range(len(self.memory)), batch_size)

    def update(*args, **kwargs):
        pass

    def __len__(self):
        return len(self.memory)

    def clean(self):
        self.memory = list()
        self.position = 0
