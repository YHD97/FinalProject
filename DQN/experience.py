import os
import random

import numpy as np


class ReplayBuffer:

    def __init__(self, size=1000, input_shape=(640, 640), history_length=4):

        self.size = size
        self.input_shape = input_shape
        self.history_length = history_length
        self.count = 0  # total index of memory written to, always less than self.size
        self.current = 0  # index to write to

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.input_shape[0], self.input_shape[1]), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        self.priorities = np.zeros(self.size, dtype=np.float32)



    def add_experience(self, action, frame, reward, terminal):

        if frame.shape != self.input_shape:
            raise ValueError('Dimension of frame is wrong!')

        # Write memory
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.priorities[self.current] = max(self.priorities.max(), 1)  # make the most recent experience important
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def get_minibatch(self, batch_size=32):

        if self.count < self.history_length:
            raise ValueError('Not enough memories to get a minibatch')

        # Get a list of valid indices
        indices = []
        for i in range(batch_size):
            while True:
                index = random.randint(self.history_length, self.count - 1)
                if index >= self.current and index - self.history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.history_length:index].any():
                    continue
                break
            indices.append(index)

        # Retrieve states from memory
        states = []
        new_states = []
        for idx in indices:
            states.append(self.frames[idx - self.history_length:idx, ...])
            new_states.append(self.frames[idx - self.history_length + 1:idx + 1, ...])

        states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1))
        new_states = np.transpose(np.asarray(new_states), axes=(0, 2, 3, 1))

        return states, self.actions[indices], self.rewards[indices], new_states, self.terminal_flags[indices]

    def set_priorities(self, indices, errors, offset=0.1):

        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def save(self, folder_name):
        """Save the replay buffer to a folder"""

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        np.save(folder_name + '/actions.npy', self.actions, allow_pickle=True,
                    fix_imports=True)
        np.save(folder_name + '/frames.npy', self.frames, allow_pickle=True,
                    fix_imports=True)
        np.save(folder_name + '/rewards.npy', self.rewards, allow_pickle=True,
                    fix_imports=True)
        np.save(folder_name + '/terminal_flags.npy', self.terminal_flags, allow_pickle=True,
                    fix_imports=True)

    def load(self, folder_name):
        """Loads the replay buffer from a folder"""
        self.actions = np.load(folder_name + '/actions.npy')
        self.frames = np.load(folder_name + '/frames.npy')
        self.rewards = np.load(folder_name + '/rewards.npy')
        self.terminal_flags = np.load(folder_name + '/terminal_flags.npy')
