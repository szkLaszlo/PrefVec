"""
This file contains the common utils for the models.
"""
import argparse
import glob
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def load_images_from_files(img_dir_path,
                           img_extention="*.jpg",
                           delete_images=True):
    img_array = []
    files = glob.glob(f'{os.path.join(img_dir_path, img_extention)}')
    files.sort(key=os.path.getmtime)

    for filename in files:
        img = cv2.imread(filename)
        img_array.append(img)
        if delete_images:
            os.remove(filename)
    return img_array


def rescale_image(img, scale_x, scale_y):
    if scale_x == 1 and scale_y == 1:
        return img
    im = np.zeros((img.shape[0] * scale_x, img.shape[1] * scale_y, 3))
    for k in range(0, im.shape[0], scale_x):
        for j in range(0, im.shape[1], scale_y):
            value = img[k // scale_x, j // scale_y]

            im[k:k + scale_x, j:j + scale_y] = value * 255 * np.ones_like(
                (im[k:k + scale_x, j:j + scale_y]))
    return im


def save_images_to_video(img_array, path, video_name, frame_rate, dim):
    if img_array.__len__() > frame_rate:
        out = cv2.VideoWriter(os.path.join(path, video_name),
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                              frame_rate, dim)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def select_activation(activation_type):
    if activation_type == "relu":
        return nn.ReLU(inplace=True)
    elif activation_type == "sigmoid":
        return nn.Sigmoid()
    elif activation_type == "softmax":
        return nn.Softmax(dim=-1)
    return None


def select_optimizer(parameters, optimizer_, learning_rate=0.0001, weight_decay=0.01):
    """
    This function selects the appropriate optimizer based on the argparser input.
    :param: args_: Contains learning rate and weight_decay attributes
    :param: model: is a torch.nn.Module with trainable parameters.
    :return: the selected optimizer
    """
    # Creating optimizer
    if optimizer_.lower() == 'asgd':
        optimizer = torch.optim.ASGD(params=parameters, lr=learning_rate,
                                     weight_decay=weight_decay)
        print('ASGD optimizer is used')
    elif optimizer_.lower() == 'adam':
        optimizer = torch.optim.Adam(params=parameters, lr=learning_rate,
                                     weight_decay=weight_decay,
                                     amsgrad=True)
        print('Adam optimizer is used')
    elif optimizer_.lower() == 'adamw':
        optimizer = torch.optim.AdamW(params=parameters, lr=learning_rate,
                                      weight_decay=weight_decay)
        print('AdamW optimizer is used')
    elif optimizer_.lower() == 'sgd':
        optimizer = torch.optim.SGD(params=parameters, lr=learning_rate)
        print('SGD optimizer is used')
    elif optimizer_.lower() == 'sparseadam':
        optimizer = torch.optim.SparseAdam(params=parameters, lr=learning_rate)
        print('SparseAdam optimizer is used')
    else:
        optimizer = torch.optim.Adam(params=parameters, lr=learning_rate,
                                     weight_decay=weight_decay)
        print('Adam optimizer is used')
    return optimizer


def plot_grad_flow(named_parameters):
    """
    This function was used to plot gradient flow
    Parameters
    ----------
    named_parameters: Names of nn.Module parameters

    Returns
    -------

    """
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


def network_grad_plot(model, writer, epoch, name=''):
    """
    Function to plot gradients of networks
    Parameters
    ----------
    model: nn.Module
    writer: SummaryWriter to write to
    epoch: defines the timestep for which this gradient is plotted

    Returns None
    -------

    """
    for name, f in model.named_parameters():
        if hasattr(f.grad, 'data'):
            hist_name = f'{name}/ + {list(f.grad.data.size())}'
            writer.add_histogram(hist_name, f.grad.data, epoch)


def network_weight_plot(model, writer, epoch):
    """
    Function to plot gradients of networks
    Parameters
    ----------
    model: nn.Module
    writer: SummaryWriter to write to
    epoch: defines the timestep for which this gradient is plotted

    Returns None
    -------

    """
    for name, f in model.named_parameters():
        if hasattr(f, 'data') and f.requires_grad:
            writer.add_histogram(name, f.data, epoch)


def find_latest_weight(path='torchSummary', file_end='.weight', exclude_end='.0'):
    """
    Function to search the latest weights in a directory.
    :param path: path to the directory.
    :param file_end: the string which will be used at the end
    :param exclude_end: this text will be excluded during search.
    :return:
    """
    if ".weight" in path and os.path.isfile(path):
        return path
    name_list = os.listdir(path)
    full_list = [os.path.join(path, i) for i in name_list]
    time_sorted_list = sorted(full_list, key=os.path.getmtime)
    for i in range(len(time_sorted_list) - 1, -1, -1):
        if not (time_sorted_list[i].endswith(file_end) or os.path.isdir(time_sorted_list[i])):
            del time_sorted_list[i]
            continue
        if len(time_sorted_list) and os.path.isdir(time_sorted_list[i]):
            latest_weight = find_latest_weight(path=time_sorted_list[i], file_end=file_end, exclude_end=exclude_end)
            if latest_weight is not None:
                return latest_weight
            else:
                del time_sorted_list[-1]
                continue
        return time_sorted_list[i]


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        print(name, "has: ", param)
        total_params += param
    print(f"Total Trainable Params: {total_params}")
    return total_params


def args_to_hparam_dict(args):
    """
    Function to convert args to hparams for tensorboard
    :param args: argumentparser instance
    :return: dict with compatible types.
    """
    hp_dict = {}
    for hpkey, hpvalue in args.__dict__.items():
        if not isinstance(hpvalue, (str, bool, int, float)):
            hp_dict[hpkey] = str(hpvalue)
        else:
            hp_dict[hpkey] = hpvalue
    return hp_dict


def str2dict(input):
    if not isinstance(input, str):
        return None
    return eval(input)


def seed_everything(seed):
    if seed is None:
        seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def manage_global_stats(stat=None, ep_info=None, is_evaluation=False):
    if stat is None:
        stat = {}
    if ep_info is not None:
        for keyx, value in ep_info.items():
            if isinstance(value, str):
                name_ = f'{keyx}_{value}'
                if name_ not in stat:
                    stat[name_] = 1
                else:
                    stat[name_] += 1
            elif value is None:
                if keyx not in stat:
                    stat[keyx] = 1
                else:
                    stat[keyx] += 1
            elif isinstance(value, (int, float)):
                if keyx not in stat:
                    stat[keyx] = value
                else:
                    stat[keyx] += value
            elif isinstance(value, (list, tuple)) and is_evaluation:
                if keyx not in stat:
                    stat[keyx] = value
                else:
                    stat[keyx] += value
    return stat


def extract_info_for_episode(info_):
    print()


def update_log_dict(dict, key, value):
    if dict is None:
        dict = {}
    if value is not None:
        if dict.get(key, None) is not None:
            dict[key].append(value)
        else:
            dict[key] = [value]

    return dict


def calculate_episode_reward(reward, w):
    if w is not None:
        ep_rew = torch.tensor(np.array(reward), dtype=torch.float32, device=torch.device("cpu")).sum(0)
        episode_reward = torch.matmul(ep_rew, w.cpu()).item()
    else:
        episode_reward = sum(reward)
    return episode_reward


def push_memory_with_discounted_rewards(memory, state_list, action_list, reward_list, cur_pref=0, gamma=0.9,
                                        immediate_reward=None, last_x=5000):
    """
    Function to push discounted returns to the memory.
    :param reward_list:
    :param gamma:
    :param action_list: batch of actions
    :param state_list: batch of states
    :param cur_pref: preference for the current actions
    :param immediate_reward: immediate reward multiplier
    :param last_x: how many samples to push maximum. This can be useful for long episodes.
    :return: None
    """

    if immediate_reward is None:
        immediate_reward = np.zeros_like(reward_list[0])

    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in reward_list[::-1]:
        R = np.array(r) + gamma * immediate_reward * R
        rewards.insert(0, R.tolist())
    # Pushing to memory
    for step_i in range(max(len(rewards) - last_x, 0), len(rewards), 1):
        memory.push(state_list[step_i].tolist(), action_list[step_i],
                    state_list[step_i + 1].tolist() if step_i < len(rewards) - 1 else None,
                    rewards[step_i], cur_pref)


class store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None:
            last_dim = int(values[0])
            values = np.array(values[1:])
            values = values.reshape((last_dim, -1))
            values = values.tolist()
        return super().__call__(parser, namespace, values, option_string)


class store_as_array2(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None:
            values = np.array(values)
            values = values.tolist()
        return super().__call__(parser, namespace, values, option_string)


class Logger:
    """
    This class is for using tensorboard to log training and evaluation metrics.
    """

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir

    def log_scalar_dict(self, log_data, step, prefix="episode", normalizer=1):
        """
        This function writes to the tensorboard with the defined dict.
        :param: log_data: Dict, containing the data to be written.
        :return: None
        """
        for key, value in log_data.items():
            if isinstance(value, list):
                self.writer.add_histogram(f"{prefix}/{key}", np.array(value), step)
            elif value is not None and not isinstance(value, str):
                self.writer.add_scalar(f"{prefix}/{key}", value / normalizer, step)

    def log_hyperparams(self, hparams):
        """
        This function writes hyperparameters to the tensorboard.
        :param: hparams: Dict, containing the hyperparameters to be written.
        :return: None
        """
        self.writer.add_hparams(convert_dict_value_to_string(hparams), {}, run_name="")

    def log_graph(self, model, input):
        pass  # self.writer.add_graph(model, input)


def convert_dict_value_to_string(in_dict):
    converted_params = {}
    for key, value in in_dict.items():
        if isinstance(value, dict):
            converted_params.update(convert_dict_value_to_string(value))
        elif not isinstance(value, (str, bool, int, float)):
            converted_params[key] = str(value)
        else:
            converted_params[key] = value
    return converted_params
