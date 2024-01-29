import copy

import numpy as np
import torch
from torch import nn
import warnings
from PrefVeC.model_prefvec.preferences import CLPreferenceSelector, DefaultPolicySelector
from PrefVeC.QN.q_agent import DQNWrapper
from PrefVeC.utils.helper_functions import create_state_action_pairs, calculate_sfs_and_get_action_values, \
    calculate_sfs_for_all_policies_and_all_actions, create_q_values, select_max_action_for_policies, \
    create_graph_representation
from PrefVeC.utils.networks import SimpleMLP, SimpleGNN


class Q(nn.Module):
    def __init__(self, network_type, input_size, actions, hidden_size=64, activation=None):
        super(Q, self).__init__()
        self.name = "Q"
        if network_type == "attention":
            from rlagents.rl_agents.agents.common.models import EgoAttentionNetwork
            self.model = EgoAttentionNetwork(config={'type': 'EgoAttentionNetwork', 'layers': [128, 128],
                                                     'embedding_layer': {'type': 'MultiLayerPerceptron',
                                                                         'layers': [64, 64], 'reshape': False,
                                                                         'in': input_size,
                                                                         'activation': 'RELU', 'out': None},
                                                     'others_embedding_layer': {'type': 'MultiLayerPerceptron',
                                                                                'layers': [64, 64],
                                                                                'reshape': False,
                                                                                'in': input_size,
                                                                                'activation': 'RELU',
                                                                                'out': None},
                                                     'self_attention_layer': None,
                                                     'attention_layer': {'type': 'EgoAttention',
                                                                         'feature_size': 64, 'heads': 2,
                                                                         'dropout_factor': 0},
                                                     'output_layer': {'type': 'MultiLayerPerceptron',
                                                                      'layers': [64, 64], 'reshape': False,
                                                                      'in': 64, 'out': actions,
                                                                      'activation': 'RELU'}, 'in': 105,
                                                     'out': actions, 'presence_feature_idx': 0})
        else:
            self.model = network_type(input_size, output_size=actions, hidden_size=hidden_size,
                                      last_layer_activation=activation)

        self.is_graph = True if network_type is SimpleGNN else False

    def forward(self, state_):
        """
        Function for model prediction
        :param state_: [batch x states]
        :return: model prediction [batch x policy x actions x SF]
        """
        if self.is_graph:
            state_ = create_graph_representation(state_)
        return self.model(state_)


class PrefVeC(nn.Module):
    """
    Model for representing the SFs one-by-one for each policy.
    """

    def __init__(self, network_type, input_size, actions, d, hidden_size=64, num_policies=6, activation=None):
        super(PrefVeC, self).__init__()
        self.name = "PrefVeC"

        self.num_policies = num_policies
        self.num_actions = actions
        self.num_sfs = d
        # Creating the Successor Features
        policies = {}
        for i in range(num_policies):
            sfs = []
            # Creating separate networks for each feature and each policy
            for j in range(d):
                if network_type == "attention":
                    from rlagents.rl_agents.agents.common.models import EgoAttentionNetwork
                    sfs.append(EgoAttentionNetwork(config={'type': 'EgoAttentionNetwork', 'layers': [128, 128],
                                                           'embedding_layer': {'type': 'MultiLayerPerceptron',
                                                                               'layers': [64, 64], 'reshape': False,
                                                                               'in': input_size,
                                                                               'activation': 'RELU', 'out': None},
                                                           'others_embedding_layer': {'type': 'MultiLayerPerceptron',
                                                                                      'layers': [64, 64],
                                                                                      'reshape': False,
                                                                                      'in': input_size,
                                                                                      'activation': 'RELU',
                                                                                      'out': None},
                                                           'self_attention_layer': None,
                                                           'attention_layer': {'type': 'EgoAttention',
                                                                               'feature_size': 64, 'heads': 2,
                                                                               'dropout_factor': 0},
                                                           'output_layer': {'type': 'MultiLayerPerceptron',
                                                                            'layers': [64, 64], 'reshape': False,
                                                                            'in': 64, 'out': actions,
                                                                            'activation': 'RELU'},
                                                           'in': 105,
                                                           'out': actions, 'presence_feature_idx': 0}))
                else:
                    sfs.append(network_type(input_size, output_size=actions, hidden_size=hidden_size,
                                            last_layer_activation=activation))
            policies[f"p{i}"] = nn.ModuleList(sfs)
        self.is_graph = True if network_type is SimpleGNN else False
        # The policies will be induced by the sf multiplication.
        self.policies = nn.ModuleDict(policies)

    def forward(self, state_):
        """
        Function for model prediction
        :param state_: [batch x states]
        :return: model prediction [batch x policy x actions x SF]
        """
        # Collecting SFs
        pol_i = []
        if self.is_graph:
            state_ = create_graph_representation(state_)
        # Calculating all SFs
        for name_, pol in self.policies.items():
            sf_list = []
            # For each policy we collect the sf values
            for sf in pol:
                # batch x 1 x action x 1  <--- the last dim is the sf value for the given policy
                sf_list.append(sf(state_))
            # Stacking the sf values together for the current policy
            pol_i.append(torch.stack(sf_list, dim=-1))

        # Stacking all policies together << batch x policy x actions x SFs
        sf_pi_j = torch.stack(pol_i, dim=1)
        # batch x policy x actions x SFs
        return sf_pi_j


class FastRL(nn.Module):
    """
    Model based on the FastRL article: https://www.pnas.org/content/117/48/30079
    """

    def __init__(self, input_size, actions, d, hidden_size=64, num_policies=6, activation=None):
        super(FastRL, self).__init__()
        self.name = "FastRL"

        self.num_policies = num_policies
        self.num_actions = actions
        self.num_sfs = d
        # Creating the Successor Features
        sfs = {}
        for i in range(d):
            sfs[f"sf{i}"] = SimpleMLP(input_size=input_size + 1, output_size=num_policies, hidden_size=hidden_size,
                                      last_layer_activation=activation)

        # The policies will be induced by the sf multiplication.
        self.sfs = nn.ModuleDict(sfs)

    def forward(self, state_):
        """
        Function for model prediction
        :param state_: [batch x states]
        :return: model prediction [batch x policy x actions x SF]
        """
        # Collecting SFs
        sf_i = []
        # (batch*actions) x features+action
        state_action_pairs = create_state_action_pairs(state_, self.num_actions)

        # Calculating all SFs
        for sf in self.sfs.values():
            # (batch*actions) x policy x 1
            out = sf(state_action_pairs)
            sf_i.append(out)

        # (batch*actions) x policy x SFs
        sf_pi_j = torch.stack(sf_i, dim=-1)
        # batch x policy x actions x SFs
        sf_pi_j = torch.stack(sf_pi_j.split(state_.size(0), dim=0), dim=2)

        return sf_pi_j


class FastRLWrapper(DQNWrapper):

    def __init__(self, model, trans, training_w, eval_w, target_model=None, target_after=10, gamma=0.99, device="cpu",
                 mix_policies=False, **kwargs):
        super(FastRLWrapper, self).__init__(model, trans,
                                            target_model=target_model,
                                            target_after=target_after,
                                            gamma=gamma,
                                            device=device)
        self.sequential = kwargs.get('sequential', False)
        self.transfer_old_policy = kwargs.get("copy_policy", False)
        self.dynamic_preference = kwargs.get("dynamic_preference", False)
        self.w = training_w.to(device=self.device)
        self.evaluation_preference = eval_w.to(device=self.device)
        self.train_preference = self.w
        self.needs_optimizer_update = False
        self.weight_loss_with_sf = kwargs.get("weight_loss_with_sf", True)

        assert self.model.num_sfs == self.train_preference().shape[0], \
            (f"The size of training preferences must match the size of the model."
             f"currently we have {self.train_preference().shape[0]}")

        assert self.evaluation_preference().shape[0] == self.train_preference().shape[0], \
            (f"Evaluation preferences must be the same size as training ones. Currently we have "
             f"{self.evaluation_preference().shape[0], self.train_preference().shape[0]}")

        self.mix_policies = mix_policies
        self.use_cumulants = True
        self.rho = kwargs.get("rho", 0.1)

    def forward_for_action(self, state_, **kwargs):
        """
        Function to retrieve the action prediction values.
        :param state_:
        :return:
        """
        greedy = kwargs["best_policy"]
        if greedy:
            j = self.train_preference.get_trained_policy_index()
        else:
            j = kwargs["j"]
        preferences = self.w(state=state_, index=kwargs.get("j", None))
        pred_action, acting_policy = calculate_sfs_and_get_action_values(model=self.model,
                                                                         state_=state_,
                                                                         preferences=preferences,
                                                                         policy_index_=j if getattr(self.w,
                                                                                                    "default_policy",
                                                                                                    None) is None else getattr(
                                                                             self.w,
                                                                             "default_policy",
                                                                             j),
                                                                         greedy=greedy if getattr(self.w,
                                                                                                  "default_policy",
                                                                                                  None) is None else False)
        return acting_policy, pred_action

    def calculate_loss_tderror(self, expected_q_value, q_value, weights, prefs=None, **kwargs):
        weights = weights.unsqueeze(-1)
        if self.mix_policies:
            weights = weights.unsqueeze(-1)
        loss = torch.sum((q_value - expected_q_value) ** 2 * weights)
        weighted_by_pref = torch.matmul(torch.abs(q_value - expected_q_value).unsqueeze(1),
                                        prefs.to(torch.float32).unsqueeze(
                                            -1)).squeeze() if self.weight_loss_with_sf else (
                q_value - expected_q_value).sum(-1)
        td_error = torch.abs(weighted_by_pref).detach()
        loss_comps = torch.mean((q_value - expected_q_value) ** 2 * weights, dim=0) / 2
        if self.mix_policies:
            td_error = td_error.sum(-1)
            loss_comps = loss_comps.sum(0)
        current_policy = self.get_preference_vector(self.get_preference_index())[0]
        logs_ = {
            "loss": loss.detach().cpu() if loss.is_cuda else loss.detach(),
            "policy_pref_0": float(current_policy)
        }
        logs_.update(
            {f"loss_comp{i}": loss_comps[i].detach().cpu() if loss_comps[i].is_cuda else loss_comps[i].detach() for i in
             range(len(loss_comps))})
        return loss, td_error, logs_

    def get_target_prediction(self, q_value, non_final_next_states, non_final_mask, rewards, policy_index_batch):

        preferences = self.w(state=non_final_next_states, index=policy_index_batch[non_final_mask])
        next_psi_t1 = torch.zeros_like(q_value, device=self.device, dtype=torch.float32, requires_grad=False)
        # added check for only terminal batch
        if sum(non_final_mask) > 0:
            with torch.no_grad():
                # batch x policy x actions x sfs
                next_sf_values = calculate_sfs_for_all_policies_and_all_actions(model=self.target_model,
                                                                                state_=non_final_next_states)
                # batch x policy x actions
                q_values = create_q_values(next_sf_values, preferences)
                # creating the indexes for best actions / policy
                max_values = torch.max(q_values, dim=-1, keepdim=True).values
                idx = q_values == max_values
                idx = idx.unsqueeze(-1).expand(next_sf_values.size())
                # batch x policy x SFs
                psi_t1 = select_max_action_for_policies(mask=idx, tensor_to_mask=next_sf_values, dim=2)

                if self.mix_policies:
                    next_psi_t1[non_final_mask] = psi_t1.detach()
                    rewards = rewards.unsqueeze(1)
                else:
                    next_psi_t1[non_final_mask] = psi_t1[
                        torch.arange(next_sf_values.size(0)), policy_index_batch[non_final_mask], ...].detach()

        # Compute the expected psi values
        expected_psi = (next_psi_t1 * self.gamma) + rewards
        return expected_psi

    def get_current_prediction(self, action_batch, policy_index_batch, state_batch):
        psi_t = calculate_sfs_for_all_policies_and_all_actions(self.model, state_batch)
        # Select applied actions from all the policies
        if self.mix_policies:
            psi_t = psi_t[torch.arange(psi_t.size(0)), :, action_batch, ...]
        else:
            psi_t = psi_t[torch.arange(psi_t.size(0)), policy_index_batch, action_batch, ...]
        return psi_t

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
        action_batch = torch.stack(batch.action).to(self.device)
        reward_batch = torch.stack(batch.reward).to(self.device)
        policy_index_batch = torch.stack(batch.index).to(self.device)

        return state_batch, action_batch, non_final_next_states, reward_batch, non_final_mask, policy_index_batch

    def get_preference_vector(self, index, is_evaluation=False):
        if is_evaluation:
            return self.evaluation_preference()
        else:
            return self.train_preference(index=index)

    def get_preference_index(self, is_evaluation=False):
        if isinstance(self.train_preference, CLPreferenceSelector):
            return self.train_preference.current_idx
        elif isinstance(self.train_preference, DefaultPolicySelector):
            return self.train_preference.__len__() - 1 if is_evaluation and self.evaluation_preference.default_policy is not None else self.train_preference.current_idx

    def set_random_preference_index(self):
        self.train_preference.set_random_preference_index()

    def get_trainable_parameters(self):
        param_list = []
        for name, param in self.model.named_parameters():
            if f"p{self.train_preference.current_idx}" in name:
                param_list.append(param)
        return param_list

    def update_optimizer(self):
        if self.needs_optimizer_update:
            self.needs_optimizer_update = False
            return True
        return False

    def eval(self):
        """
        Function to set the model to eval mode
        """

        self.model.eval()
        self.w = self.evaluation_preference

    def train(self):
        """
        Function to set the model to train mode.
        """
        self.model.train()
        self.w = self.train_preference

    def copy_policy(self, loss, memory):
        if isinstance(self.w, CLPreferenceSelector) and self.sequential:
            if self.w.update(current_loss=loss.detach().cpu()):
                if self.dynamic_preference:
                    # calculate the required value for 20% action selection change
                    value_change_in_pref = self.calculate_min_changes(mem=memory)
                    # set the value
                    self.w.set_new_value(value_change_in_pref, dim=0)
                if self.transfer_old_policy:
                    new_state_dict = {}
                    for keyx, val in self.model.state_dict().items():
                        if f"p{self.w.current_idx - 1}" in keyx:
                            name_ = str(keyx).replace(f"p{self.w.current_idx - 1}", f"p{self.w.current_idx}")
                            new_state_dict[name_] = copy.deepcopy(val)
                    self.model.load_state_dict(new_state_dict, strict=False)
                    # turning off gradients for old parts of the model
                    for keyx, param in self.model.named_parameters():
                        if f"p{self.w.current_idx - 1}" in keyx:
                            param.requires_grad = False
                            param.grad = None
                self.needs_optimizer_update = True
                memory.save(name=f"memory_policy_{self.w.current_idx - 1}")
                memory.reset()
                return True

        return False

    def calculate_min_changes(self, mem):
        buffer = getattr(mem, "common_buffer", None)
        if buffer is None:
            random_batch, _, _ = mem.sample(50000 if len(mem) > 50000 else len(mem))
        else:
            random_batch, _, _ = buffer.sample(50000 if len(buffer) > 50000 else len(buffer))
        batch = self.trans(*zip(*random_batch))
        state_batch, action_batch, non_final_next_states, reward_batch, \
            non_final_mask, policy_index_batch = self.extract_batch_data(batch)
        # ensure to check the latest policy
        policy_index_batch = torch.ones_like(policy_index_batch, dtype=torch.long) * (
                self.w.get_trained_policy_index() - 1)
        with torch.no_grad():
            old_preferences = self.w(state=state_batch, index=policy_index_batch)
            # batch x policy x actions x sfs
            next_sf_values = calculate_sfs_for_all_policies_and_all_actions(model=self.model, state_=state_batch)
            # batch x policy x actions
            old_q_values = create_q_values(next_sf_values, old_preferences)
            # helper for indexing batch dimension
            batch_size = torch.arange(old_q_values.size(0))
            # selecting the currently best actions and their indexes
            x, x_idx = old_q_values[batch_size, policy_index_batch].max(-1)
            # creating the sf values for each action
            sf_11 = next_sf_values[batch_size, policy_index_batch, x_idx, 0]
            sf_12 = next_sf_values[batch_size, policy_index_batch, x_idx, 1]
            sf_21 = next_sf_values[batch_size, policy_index_batch, :, 0]
            sf_22 = next_sf_values[batch_size, policy_index_batch, :, 1]
            # calculating the difference between the first and second action sfs
            kl = sf_11.unsqueeze(-1) - sf_21
            # calculating the z values based on the inequality
            z = (kl * old_preferences[:, 0].unsqueeze(-1) + sf_12.unsqueeze(-1) - sf_22) / -kl
            # removing nans due to zero division
            z[z.isnan()] = np.inf
            # get the min and sort the values of sf changes
            list_of_diffs = sorted(z.min(-1).values.tolist())
            min_value_to_change = self.select_min_value_from_list(list_of_diffs)

        return min_value_to_change

    def select_min_value_from_list(self, list_of_diffs):
        # select min change value based on the expected self.rho % action change
        min_value_to_change = list_of_diffs[int(len(list_of_diffs) * self.rho) + 1]
        if min_value_to_change > 0:
            idx = 1
            for idx, min_ in enumerate(list_of_diffs):
                if min_ > 0:
                    warnings.warn(f"Could not find a value to change with the given rho value."
                                  f"Selecting {idx/len(list_of_diffs)}% = {list_of_diffs[idx-1]} weight change")
                    break

            min_value_to_change = list_of_diffs[idx - 1]
        return min_value_to_change


class CLDQNWrapper(DQNWrapper):
    def __init__(self, model, trans, training_w, eval_w, env, target_model=None, target_after=10, gamma=0.99,
                 device="cpu",
                 mix_policies=False, **kwargs):
        super(CLDQNWrapper, self).__init__(model, trans,
                                           target_model=target_model,
                                           target_after=target_after,
                                           gamma=gamma,
                                           device=device)
        self.w = training_w.to(device=self.device)
        self.evaluation_preference = eval_w.to(device=self.device)
        self.train_preference = self.w
        self.env = env

        assert self.env.default_w.size == self.train_preference().shape[0], \
            (f"The size of training preferences must match the size of the model."
             f"currently we have {self.train_preference().shape[0]}")

        assert self.evaluation_preference().shape[0] == self.train_preference().shape[0], \
            (f"Evaluation preferences must be the same size as training ones. Currently we have "
             f"{self.evaluation_preference().shape[0], self.train_preference().shape[0]}")

        self.use_cumulants = False

    def get_preference_vector(self, index, is_evaluation=False):
        return None

    def get_preference_index(self, is_evaluation=False):
        return self.train_preference.current_idx

    def eval(self):
        """
        Function to set the model to eval mode
        """

        self.model.eval()
        self.w = self.evaluation_preference
        self.env.unwrapped.default_w = self.w(index=self.w.current_idx).numpy()

    def train(self):
        """
        Function to set the model to train mode.
        """
        self.model.train()
        self.w = self.train_preference
        self.env.unwrapped.default_w = self.w(index=self.w.current_idx).numpy()

    def copy_policy(self, loss, memory):
        if isinstance(self.w, CLPreferenceSelector):
            self.w.update(current_loss=loss.detach().cpu())
        return False
