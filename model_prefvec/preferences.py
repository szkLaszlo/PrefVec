import numpy as np
import torch
from torch import nn


class DefaultPolicySelector(nn.Module):
    """
    This is a basic SF preference selector which selects a given preference from a set of predefined preferences.
    The actually selected values are selected random, or if the default policy is a number,
     that will be selected all the time.
    """

    def __init__(self, w, default_policy=None, is_evaluation=False):
        super(DefaultPolicySelector, self).__init__()
        self.w = torch.tensor(w, dtype=torch.float32, requires_grad=False)
        assert self.w.ndim == 2
        self.default_policy = default_policy
        self.current_idx = 0 if not is_evaluation else len(w) - 1

    def forward(self, state=None, index=0):
        if isinstance(index, int) and len(self) - 1 < index:
            return self.w[0]
        elif isinstance(index, torch.Tensor) and all(len(self) - 1 < index):
            return self.w[torch.zeros_like(index)]
        return self.w[index]

    def __len__(self):
        return self.w.size(0)

    def set_random_preference_index(self):
        max_pref_id = len(self)
        if max_pref_id > 0:
            self.current_idx = np.random.randint(max_pref_id) if self.default_policy is None else self.default_policy

    def update(self, **kwargs):
        return False

    def to(self, device):
        self.w = self.w.to(device)
        return self

    def get_trained_policy_index(self):
        return self.__len__() - 1


class CLPreferenceSelector(nn.Module):
    """
    This preference selector is used to continually change the policies.
    A current policy is selected based on the update times or loss standard deviation.
    """

    def __init__(self, weights, scheduler_steps, is_evaluation=False, average_range=1000):
        super(CLPreferenceSelector, self).__init__()
        self.scheduler_steps = scheduler_steps
        self.w = torch.tensor(weights, dtype=torch.float32, requires_grad=False)
        self.current_idx = 0 if not is_evaluation else len(weights) - 1
        self.loss_queue = []
        self.average_range = average_range
        self.current_update = 0

    def forward(self, state=None, index=0) -> torch.Tensor:
        return self.w[index]

    def __len__(self) -> int:
        return self.w.size(0)

    def update(self, current_loss):

        self.current_update += 1
        if (self.current_update % self.scheduler_steps == 0 and self.current_update > 2 and self.w.size(
                0) - 1 > self.current_idx):
            self.current_idx += 1
            self.current_update = 0
            return True

        return False

    def set_random_preference_index(self):
        pass

    def to(self, device):
        self.w = self.w.to(device)
        return self

    def get_trained_policy_index(self):
        return self.current_idx

    def set_new_value(self, change_val, dim):
        if change_val < 0:
            if self.current_idx != self.w.size(0) - 1 and self.current_idx > 0:
                self.w[self.current_idx][dim] = max(float(self.w[self.current_idx - 1][dim] + change_val),
                                                    float(self.w[-1][dim]))
        else:
            self.current_idx = max(0, self.current_idx - 1)
