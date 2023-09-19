"""
@author "Laszlo Szoke" 

"""
import numpy as np
import torch
from torch import nn


def get_normalized_action_vector(batch_size, dtype, device, num_actions: int):
    """
    Function to normalize the action vector.
    :param batch_size: batch size of the actions
    :param dtype: type of the actions to be represented in
    :param device: device to put the tensor to
    :param num_actions: how many actions are there
    :return: [batch x actions x 1]
    """
    actions = torch.arange(num_actions, dtype=dtype, device=device).unsqueeze(0).expand((batch_size, -1)).unsqueeze(-1)
    action_size = int(actions.size(-2) - 1)
    actions = actions / action_size
    return actions


def calculate_sfs_for_all_policies_and_all_actions(model: nn.Module, state_: torch.Tensor):
    """
    This function calculates the successor features for all policies and actions
    :param model: nn.Module to predict sfs
    :param state_: batch x features
    :return: batch x policies x actions x SFs
    """

    # calculate SFs for the current state
    # batch x policies x actions x SFs
    sfs = model(state_)

    return sfs


def argmax(q_values):
    """
    Takes in a list of q_values and returns the index of the item
    with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in q_values
    """
    top_value = float("-inf")
    ties = []
    assert q_values.shape[0] == 1
    q_values = q_values.squeeze(0)
    for i in range(len(q_values)):
        # if a value in q_values is greater than the highest value update top and reset ties to zero
        if q_values[i] > top_value:
            ties = [i]
            top_value = float(q_values[i])
            # if a value is equal to top value add the index to ties
        elif q_values[i] == top_value:
            ties.append(i)
            # return a random selection from ties.
    return np.random.choice(ties)


def distance(vehicle1, vehicle2):
    x1, y1 = vehicle1.position
    x2, y2 = vehicle2.position
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def calculate_sfs_and_get_action_values(model, state_: torch.Tensor, preferences, policy_index_, greedy=False):
    """
    This function is to calculate the successor features and get the action values.
    :param model: nn.Module to approximate the SFs
    :param state_: batch x features
    :param preferences: preference vector to calculate w conditioned Q values
    :param policy_index_: policies to select from the available
    :param greedy: if True, we select the maximum over the policies
    :return: batch x actions
    """
    psi = calculate_sfs_for_all_policies_and_all_actions(model, state_)
    next_values, acting_policy = get_action_values(sfs=psi, w=preferences,
                                                   policy_index=policy_index_, greedy=greedy)

    assert next_values.size(0) == state_.size(0)

    return next_values, acting_policy


def create_state_action_pairs(state_: torch.Tensor, num_actions: int):
    """
    Function which concatenates states with actions and stacks it over the batch dimension for more efficient calculation
    :param state_: batch x features
    :param num_actions: int number of actions used for the model.
    :return: (batch*num_actions) x features+1
    """
    # create action vector for concatenation with current state
    actions = get_normalized_action_vector(batch_size=state_.size(0), dtype=state_.dtype, device=state_.device,
                                           num_actions=num_actions)

    # Extending state with the actions for simplified forward pass
    extended_input = state_.unsqueeze(1).expand((state_.size(0), actions.size(1), -1))
    temp_state_action_pairs = torch.cat((extended_input, actions), dim=-1)
    state_action_pairs = torch.cat(temp_state_action_pairs.split(1, dim=1), dim=0).squeeze(1)

    # (batch*num_actions) x features+1
    return state_action_pairs


def format_index_vector(index, device):
    """
    Function to format the index vector into a tensor
    :param index: indexes
    :param device: the device to put the tensor to
    :return: [batch]
    """

    if getattr(index, "ndim", 0) != 1:
        index = torch.tensor([index], requires_grad=False, device=device)
    return index


def create_q_values(sfs: torch.Tensor, w):
    """
    Function to calculate Q values based on the SFs and preference vectors.
    :param sfs: successor features [batch x policies x actions x SFs]
    :param w: preference vector to induce Q values
    :return: [batch x policies x actions]
    """
    w_prepared = w if w.ndim == 1 else w.unsqueeze(1).expand((sfs.size(0), sfs.size(1), sfs.size(-1))).unsqueeze(-1)
    q_val = torch.matmul(sfs, w_prepared).squeeze(-1)

    return q_val


def select_actions_over_policies(q_values, index, greedy):
    """
    Function to select actions from the available policies.
    :param q_values: Q values with size [batch x policies x actions]
    :param index: the policies to select [batch]
    :param greedy: If True, the maximum values will be selected over the policies
    :return: Q values with size [batch x actions]
    """

    if greedy:
        # selecting only the policies we have trained so far
        if index.ndim == 1:
            q_values = q_values[torch.arange(q_values.size(0)), :index + 1]
        # Q_max per actions max over the policies [batch x actions]
        q_max_a, acting_policy = torch.max(q_values, 1)

    else:
        # batch x policies x actions --> batch x actions
        # if the policies are given, we select those. (useful for next_value selection)
        q_max_a = q_values[torch.arange(q_values.size(0)), index]
        acting_policy = index

    return q_max_a, acting_policy


def create_naming_convention(args):
    # setting folder name for env
    model_name = f"{args.env_model}_{args.type_as}/{args.comment + '/' if args.comment else ''}"

    # setting name of training model
    if args.use_double_model:
        model_name += "Double"
    if args.model_version in "sf":
        model_name += "FastRL"
    elif args.model_version in "cl":
        model_name += "PrefVeC"
    elif args.model_version in "q":
        model_name += "Q"
    else:
        raise NotImplementedError

    # setting name for training method
    if args.model_train_type in "q":
        model_name += ""
    elif args.model_train_type in "sequential":
        model_name += "Sequential"
        if args.copy_policy:
            model_name += "_copy"
    elif args.model_train_type in "dynamic":
        model_name += "Dynamic"
    elif args.model_train_type in "parallel":
        model_name += "Parallel"
    elif args.model_train_type in "contiQ":
        model_name += "Curriculum"
    else:
        raise NotImplementedError

    return model_name


def get_action_values(sfs: torch.Tensor, w, policy_index=None, greedy=False):
    """
    Gets the action values for the current successor features
    :param sfs: output of the FastRL. dim: [batch x policies x actions x features]
    :param w: preference vector to use for the action value generation
    :param policy_index: batch of selected policies dim: [batch]
    :param greedy: If True, the max action will be selected
    :return: batch x actions
    """

    policy_index = format_index_vector(policy_index, sfs.device)

    assert policy_index.ndim == 1
    assert sfs.ndim == 4

    # Calculate q values from sfs and preferences
    # [batch x policies x actions]
    q_val = create_q_values(sfs, w)

    # Select final Q values from the policies
    # [batch x actions]
    q_max_a, acting_policy = select_actions_over_policies(q_values=q_val, index=policy_index, greedy=greedy)
    # batch x actions
    return q_max_a, acting_policy


def limit_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.grad.data.clamp_(-1, 1)


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def select_max_action_for_policies(mask, tensor_to_mask, dim):
    masked = torch.mul(tensor_to_mask, mask)
    neg_inf = torch.zeros_like(tensor_to_mask)
    neg_inf[~mask] = -np.inf  # Place the smallest values possible in masked positions
    return (masked + neg_inf).max(dim=dim)[0]


def soft_update_q_network_parameters(target: nn.Module,
                                     online: nn.Module,
                                     alpha: float) -> None:
    """In-place, soft-update of target parameters with parameters from online."""
    for p1, p2 in zip(target.parameters(), online.parameters()):
        p1.data.copy_(alpha * p2.data + (1 - alpha) * p1.data)


def synchronize_q_networks(target: nn.Module, online: nn.Module) -> None:
    """In place, synchronization of target and online."""
    _ = target.load_state_dict(online.state_dict())


def search_similar_key_in_dict(dict_to_search, key_to_search, default=None):
    similar_keys = [keyx for keyx in dict_to_search.keys() if key_to_search in keyx]
    if len(similar_keys) > 0:
        return dict_to_search[similar_keys[0]]
    else:
        return default


def create_graph_representation(batch):
    # reshape to [batch x nodes x features]
    a = batch.reshape(batch.size(0), -1, 5)
    # extract features
    features = a[:, :, 1:5]
    # create mask for presence of nodes
    mask = a[:, :, 0] > 0
    # select existing nodes and filter their features
    feature_mask = features[:, :, 0][mask]
    # collect indices of existing nodes
    idxs = torch.arange(feature_mask.flatten().shape[0])
    # create mask for ego nodes
    egomask = torch.ones_like(mask)
    egomask[:, 1:] = 0
    # select ego ids
    ego_ids = idxs[egomask[mask]]
    # create link to ego node
    stop_nodes = ego_ids.unsqueeze(-1).expand_as(mask)[mask]
    # create graph features
    x1 = features[mask]
    # create graph edges
    edge_index1 = torch.stack([idxs, stop_nodes])
    # filter ego time
    ego_time1 = batch[:, 0]

    return x1, edge_index1, egomask, ego_time1, ego_ids


def create_graph_rep(batch):
    start_node_list = []
    end_node_list = []
    node_feature_matrix = []
    batch_list = []
    ego_time = []
    current_ego_id = 0
    ego_idx = []
    for b_num, batch_i in enumerate(batch):
        node_feature_matrix.extend([batch_i[1:5]])
        start_node_list.extend([current_ego_id])
        end_node_list.extend([current_ego_id])
        batch_list.append(b_num)
        ego_idx.append(True)
        i = 1
        ego_time.append(batch_i[0])
        num_nodes = 1
        while i < len(batch_i) / 5:
            if batch_i[i * 4 + 1] > 0:
                node_feature_matrix.append(batch_i[i * 4 + 2:i * 4 + 6])
                start_node_list.extend([current_ego_id + num_nodes])
                end_node_list.extend([current_ego_id])
                batch_list.append(b_num)
                ego_idx.append(False)
                num_nodes += 1
            i += 1
        current_ego_id += num_nodes - 1

    # Creating the graph connectivity
    edge_index = torch.tensor([start_node_list,
                               end_node_list], dtype=torch.long)

    # Node feature matrix with shape [num_nodes, num_node_features]
    x = torch.stack(node_feature_matrix)

    # Determines which node belongs to which graph
    batch = torch.tensor(batch_list).long()
    # Creating the graph
    return x, edge_index, batch, torch.stack(ego_time), torch.tensor(ego_idx, dtype=torch.bool)
