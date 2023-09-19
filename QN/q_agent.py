"""
This script shows the QN agent related functions and wrappers.
"""

import numpy as np
import torch

from PrefVeC.utils.helper_functions import argmax, limit_gradients, synchronize_q_networks
from PrefVeC.utils.wrapper import ModelWrapperInterface


class DQNWrapper(ModelWrapperInterface):
    def __init__(self, model, trans, target_model=None, target_after=10, gamma=0.99, device="cpu", **kwargs):
        super().__init__()
        self.name = f"{'Double' if target_model is not None else ''}{model.name}"
        self.device = torch.device(device)
        self.model = model.to(device=self.device)
        self.double_model = True if target_model is not None else False
        self.target_model = target_model.to(device=self.device) if self.double_model else self.model
        self.gamma = gamma
        self.trans = trans
        self.target_after = target_after
        self.target_model.eval()
        self.use_cumulants = False
        self.current_update = 0

    def forward(self, state_, **kwargs):
        """
        This method makes the forward pass of the FastRL wrapper. It selects an action.
        :param state_: State to predict from
        :return: Chosen action based on input and the selected policy
        """

        state_ = torch.tensor(state_, dtype=torch.float, requires_grad=False, device=self.device)
        best_action = kwargs.get("best_action", False)

        with torch.no_grad():

            acting_policy, pred_action = self.forward_for_action(state_, **kwargs)

            # Selecting action or exploring
            if best_action:
                # batch x actions
                action_ = argmax(pred_action)
            else:
                action_ = np.random.randint(low=pred_action.size(-1), size=(1,))
        return action_, acting_policy

    def forward_for_action(self, state_, **kwargs):
        pred_action = self.model(state_)
        acting_policy = None
        return acting_policy, pred_action

    def update(self, optimizer, memory, batch_size, **kwargs):
        """
        This function solves the network parameter update of the agent.
        :param optimizer: an optimizer that we use to update the parameters.
        :param batch_size:
        :param memory:
        :return: loss of the update
        """

        if not len(memory) >= batch_size:
            return {}

        transitions, weights, tree_idxs = memory.sample(batch_size)
        batch = self.trans(*zip(*transitions))
        weights = torch.tensor(np.array(weights), device=self.device, dtype=torch.float32, requires_grad=False)

        state_batch, action_batch, non_final_next_states, reward_batch, \
        non_final_mask, policy_index_batch = self.extract_batch_data(batch)

        # We calculate the q_max values for the policies, then select the psi values which caused the q_max actions.
        q_value = self.get_current_prediction(state_batch=state_batch, action_batch=action_batch,
                                              policy_index_batch=policy_index_batch)

        # The future values are also calculated, then the best actions based on the current policy are calculated.
        # After this, the future psi values are created.
        expected_q_value = self.get_target_prediction(q_value=q_value,
                                                      non_final_next_states=non_final_next_states,
                                                      non_final_mask=non_final_mask,
                                                      rewards=reward_batch,
                                                      policy_index_batch=policy_index_batch,
                                                      )
        preferences = self.get_preference_vector(index=policy_index_batch)
        loss, td_error, logs = self.calculate_loss_tderror(expected_q_value, q_value, weights, prefs=preferences)
        collisions = memory.collisions_in_batch(tree_idxs)
        logs.update({"collisions_seen": collisions})
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # norm them between -1 and 1
        limit_gradients(self.model)

        optimizer.step()

        self.current_update += 1

        memory.update_priorities(tree_idxs, td_error.numpy() if not td_error.is_cuda else td_error.cpu().numpy())

        is_policy_copied = self.copy_policy(loss, memory)

        if self.double_model and (self.current_update % self.target_after == 0 or is_policy_copied):
            synchronize_q_networks(target=self.target_model, online=self.model)

        return logs

    def copy_policy(self, loss, memory):
        return False

    def calculate_loss_tderror(self, expected_q_value, q_value, weights, **kwargs):
        loss = torch.sum((q_value - expected_q_value) ** 2 * weights)
        td_error = torch.abs(q_value - expected_q_value).detach()
        logs_ = {
            "loss": loss.detach().cpu() if loss.is_cuda else loss.detach(),
        }
        return loss, td_error, logs_

    def get_current_prediction(self, action_batch, policy_index_batch, state_batch):
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        state_action_values = self.model(state_batch).gather(1, action_batch).squeeze()
        return state_action_values

    def get_target_prediction(self, q_value, non_final_next_states, non_final_mask, rewards, policy_index_batch):
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.

        next_state_values = torch.zeros(rewards.size(0), device=self.device, dtype=torch.float32)
        # applying safety check for all terminal batch
        if sum(non_final_mask) > 0:
            next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + rewards
        return expected_state_action_values

    def extract_batch_data(self, batch):

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: not s.isnan().sum(),
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        if non_final_mask.sum():
            non_final_next_states = torch.stack(
                [s for idx, s in enumerate(batch.next_state) if non_final_mask[idx].item()]).to(self.device)
        else:
            non_final_next_states = torch.tensor([], device=self.device, dtype=torch.float32)
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.stack(batch.action).unsqueeze(-1).to(self.device)
        reward_batch = torch.stack(batch.reward).to(self.device)

        return state_batch, action_batch, non_final_next_states, reward_batch, non_final_mask, None

    def create_transition(self, **kwargs):
        trans_values = []
        for k in self.trans._fields:
            tensor_ = torch.tensor(kwargs.get(k), device=self.device)
            trans_values.append(tensor_)
        trans = self.trans(*trans_values)
        return trans

    def get_preference_vector(self, index, is_evaluation=False):
        return None

    def get_preference_index(self, is_evaluation=False):
        return 0

    def set_random_preference_index(self):
        pass

    def get_saliency_map(self, state_, **kwargs):
        state_ = torch.tensor(state_, dtype=torch.float, requires_grad=True, device=self.device)

        with torch.set_grad_enabled(True):
            _, pred_action = self.forward_for_action(state_, **kwargs)
            # batch x actions
            max_value, action_ = pred_action.max(-1)
            max_value.backward()
        slc = torch.abs(state_.grad[0].cpu())
        slc = (slc - slc.min()) / (slc.max() - slc.min())

        return slc.numpy().tolist()

    def get_trainable_parameters(self):
        return self.model.parameters()

    def update_optimizer(self):
        return False

    def eval(self):
        """
        Function to set the model to eval mode
        """
        self.model.eval()

    def train(self):
        """
        Function to set the model to train mode.
        """
        self.model.train()
