"""
Created by Laszlo Szoke based on https://github.com/Howuhh/prioritized_experience_replay/
"""
import os
import pickle
import random
from collections import namedtuple

import numpy as np
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
TransitionFastRL = namedtuple('Transition',
                              ('state', 'action', 'next_state', 'reward', 'index'))


class SumTree:
    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1  # child index in tree array
        change = value - self.nodes[idx]

        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2 * idx + 1, 2 * idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]

    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"


class ReplayMemory(object):
    """
    Class for Replay Memory for the DQN agents.
    """

    def __init__(self, capacity, terminal_percentage=None):
        """
        Initializer for the class.
        :param capacity: defines how many transitions to keep in the memory.
        """
        self.capacity = capacity
        self.memory = []
        self.terminal_percentage = terminal_percentage

    def push(self, trans, **kwargs):
        """
        Saves a transition to the replay memory.
        :param args: should contain state, reward, done, info
        """
        if len(self.memory) >= self.capacity:
            i = 0
            if self.terminal_percentage is not None:
                # searching for the terminal episodes
                while self.memory[i].next_state is None and i < self.capacity * self.terminal_percentage:
                    i += 1
                # selecting a random terminal episode to drop if the memory contains more than
                if i >= self.capacity * self.terminal_percentage:
                    i = random.randint(0, i)
            self.memory.pop(i)

        self.memory.append(trans)

    def sample(self, batch_size):
        """
        The function returns a batch_size of the saved transitions.
        :param batch_size: defines how many samples we want.
        :return: batch of transitions.
        """
        return random.sample(self.memory, batch_size), np.ones((batch_size,)), None

    def update_priorities(self, data_idxs, priorities):
        pass

    def __len__(self):
        """
        Gets the length of the memory replay.
        :return: length of the memory
        """
        return len(self.memory)

    def reset(self):
        self.memory = []

    def collisions_in_batch(self, idxs):
        pass

    def collisions_in_buffer(self):
        pass


class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, eps=1e-2, alpha=0.6, beta=0.4):
        self.tree = SumTree(size=buffer_size)
        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        # transition: state, action, reward, next_state, done
        self.memory = [None] * buffer_size
        self.is_rare_idx = [None] * buffer_size

        self.count = 0
        self.real_size = 0
        self.capacity = buffer_size

        self.save_path = None
        self.current_file_idx = 0

    def reset(self):
        self.tree = SumTree(size=self.capacity)
        self.memory = [None] * self.capacity
        self.is_rare_idx = [None] * self.capacity
        self.count = 0
        self.real_size = 0

    def push(self, trans, is_rare=False, **kwargs):

        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)

        # store transition in the buffer
        self.memory[self.count] = trans
        self.is_rare_idx[self.count] = is_rare

        # update counters
        self.count = (self.count + 1) % self.capacity
        self.real_size = min(self.capacity, self.real_size + 1)

        if self.count == 0:
            self.save(name=f'full_buffer{self.current_file_idx}')
            self.current_file_idx += 1

    def sample(self, batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = []

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities.append(priority)
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = np.array(priorities) / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (self.real_size * probs) ** -self.beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates.
        # (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()

        batch = [self.memory[idx] for idx in sample_idxs]
        assert set(sample_idxs) == set(tree_idxs), "sample_idxs and tree_idxs should be the same"
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def collisions_in_batch(self, idxs):
        return sum([self.is_rare_idx[idx] for idx in idxs]) / len(idxs)

    def collisions_in_buffer(self):
        return sum(self.is_rare_idx[:self.real_size]) / self.real_size

    def save(self, name=""):
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            with open(self.save_path + f"{name}.rmd", 'wb') as f:
                pickle.dump([[val for val in self.memory[idx]] for idx in range(self.real_size)], f)

    def __len__(self):
        """
        Gets the length of the memory replay.
        :return: length of the memory
        """
        return self.real_size
