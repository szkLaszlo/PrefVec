"""
@author "Laszlo Szoke" 

"""
import glob
import json
import os
import pickle

import numpy as np

from continuousSUMO.evaluation import eval_full_statistics, decode_w_for_readable_names


def load_evaluation_statistics_from_folder(path_to_env_log, extention="*.pkl"):
    """
    Function to collect all logs of the episodes
    :param path_to_env_log: path to the directory of the env logs
    :param extention: the file ending
    :return: statistics of the folder
    """
    files = glob.glob(f'{os.path.join(path_to_env_log, extention)}')
    files.sort(key=os.path.getmtime)
    with open(f'{os.path.split(path_to_env_log)[0]}/args_eval.txt', 'r') as f:
        params = json.load(f)
    statistics_in_folder = []
    for filename in files:
        return_ = load_episode_stat(filename)
        statistics_in_folder.append(return_)
    return statistics_in_folder


def load_episode_stat(file):
    """
    Function to read and collect the data from file
    :param file: path to the data
    :return: dict of interesting attributes
    """
    with open(file, "br") as f:
        dict_ = pickle.load(f)

    return dict_

def select_state_and_action_from_dict(episode_dict):
    state = np.asarray(episode_dict["state"])[:-1, -1]
    action = np.asarray(episode_dict["action"])
    return np.stack([state, action], -1)

def plot_state_action_distribution():
    pass


def plot_action_histograms(dir_stat, cur_dir):
    import seaborn as sns
    state_action = [select_state_and_action_from_dict(s) for s in dir_stat]
    a = np.concatenate(state_action)
    data = [a[:, 0][a[:, 1] == i] for i in range(3)]
    for action in range(3):
        sns.distplot(data[action], hist=False, kde=True, kde_kws={"shade": True}, norm_hist=True, label=action)
    plt.legend()
    plt.title(cur_dir)
    plt.show()


def plot_episodic_rewards(dir_stat, cur_dir):
    import seaborn as sns
    rewards = [np.asarray(s["reward"]).sum(0) for s in dir_stat]
    a = np.stack(rewards)
    for sf in range(a.shape[-1]):
        sns.distplot(a[:, sf], hist=False, kde=True, kde_kws={"shade": True}, norm_hist=True, label=sf)
    plt.legend()
    plt.title(cur_dir)
    plt.show()


def get_collision_per_km(dir_stat):
    print()
    kms = 0
    collisions = 0
    for epi in dir_stat:
        if "collision" in epi["cause"]:
            collisions += 1

        kms += (epi['state'][-1][-2] - epi['state'][0][-1]) * 0.352

    return collisions/kms


if __name__ == "__main__":
    dir_of_eval = [
        "/cache/cl/changed_env/DoubleFastRLv1_SuMoGyM_discrete_longitudinal/20220211_095051", #3policy
    ]
    import matplotlib.pyplot as plt
    for run in dir_of_eval:
        global_stat = []
        eval_dirs = os.listdir(run)
        eval_dirs.sort()
        for dir_ in eval_dirs:
            if "eval" not in dir_:
                continue
            single_dir_stat = load_evaluation_statistics_from_folder(os.path.join(run, dir_, "env"))
            plot_action_histograms(dir_stat=single_dir_stat, cur_dir=dir_)
            plot_episodic_rewards(dir_stat=single_dir_stat, cur_dir=dir_)
            # get_collision_per_km(single_dir_stat)
            # episodic return
            global_stat.append(single_dir_stat)





