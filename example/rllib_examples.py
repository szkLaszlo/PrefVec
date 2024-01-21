"""
This file contains code for running our environments with rllib.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from typing import Dict

import numpy as np
import ray
import torch
from numpy.distutils.fcompiler import str2bool
from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.evaluation import Episode

from PrefVeC.other_envs.grid import GridWorld
from PrefVeC.other_envs.highway_cl import CumulantIntersectionEnv
from PrefVeC.utils.utils import store_as_array


def train_algo(algo, exit_criteria="episode_reward_mean", exit_threshold=0.95, max_iterations=None, verbose=False):
    best_checkpoint = None
    max_reward = - np.inf
    current_iteration = 0
    result = {"evaluation": {exit_criteria: -1.0}}
    current_checkpoint = None
    while result["evaluation"].get(exit_criteria, 0.) < exit_threshold:
        result.update(algo.train())
        current_iteration = result["episodes_total"]
        if verbose:
            print(f"Iteration: {result['training_iteration']} Reward: {result[exit_criteria]},"
                  f" evaluation {exit_criteria}: {result['evaluation'].get(exit_criteria, 0.)} ")
        if result.get("evaluation", {}).get(exit_criteria, -1.) > max_reward:
            if best_checkpoint is not None:
                algo.delete_checkpoint(best_checkpoint)
            best_checkpoint = algo.save()
            max_reward = result.get("evaluation", {}).get(exit_criteria, 0.)
        else:
            if current_checkpoint is not None:
                algo.delete_checkpoint(current_checkpoint)
            current_checkpoint = algo.save()

        if max_iterations is not None and current_iteration >= max_iterations:
            print(f"Maximum number of episodes reached: {max_iterations}")
            break
    checkpoint_dir = algo.save()
    print(f"Checkpoint saved in directory {checkpoint_dir}")
    return algo


class MyCallbacks(DefaultCallbacks):
    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        if hasattr(episode, "last_info_for"):
            info = episode.last_info_for()
            cause = info['cause']
        else:
            cause = episode._last_infos["agent0"]["cause"]
            if cause is None:
                assert episode._last_terminateds['agent0']
                cause = "success"
        cause_list = ["collision", "slow", "success"]
        for cause_i in cause_list:
            if cause_i in cause:
                episode.custom_metrics[f"{cause_i}"] = 1
            else:
                episode.custom_metrics[f"{cause_i}"] = 0


def custom_log_creator(custom_path, custom_str):
    from ray.tune.logger import UnifiedLogger

    timestr = datetime.today().strftime("%m%d_%H%M_")
    logdir_prefix = f"{custom_str}_{timestr}"

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def select_env(args):
    if args.env_model == "grid":
        n = 10
        env = GridWorld
        env_config = {
            "n": n,
            "default_w": args.default_w,
            "offscreen_rendering": True,
            "video": False,
            "grid_type": "semi_static",
            "obs_type": "grid",
            "use_step_reward": args.use_step_reward}
        return env, env_config

    elif args.env_model == "intersection":
        env_config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ['presence', 'x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],

                },
                "see_behind": True,
                "absolute": True,
                "flatten": True,
                # "normalize": True,
                "observe_intentions": False,
                "order": "shuffled",
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": False,
                "acceleration_range": [-6, 3],
                "target_speeds": [0, 4.5, 9]
            }, "controlled_vehicles": 1, "policy_frequency": args.policy_freq,
            "initial_vehicle_count": args.init_vehicles,
            "simulation_frequency": args.sim_freq,
            "spawn_probability": args.spawn_prob,
            "duration": 13, 'collision_reward': 1, 'high_speed_reward': 1, 'arrived_reward': 1,
            'reward_speed_range': [7.0, 9.0], 'normalize_reward': False,
            "offscreen_rendering": True, "video": False,
            "default_w": args.default_w,
            "go_straight": args.go_straight,
        }

        env = CumulantIntersectionEnv
        return env, env_config


def save_training_args(args, save_path, add_git_hash=True):
    if add_git_hash:
        # Get the current git commit hash
        result_hash = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
        commit_hash = result_hash.stdout.decode().strip()
        setattr(args, "commit_hash", commit_hash)
        result = subprocess.run(['git', 'diff'], stdout=subprocess.PIPE)
        commit_diff = result.stdout.decode().strip()
        setattr(args, 'commit_diff', commit_diff)

    # Saving hyperparameters
    with open(f'{os.path.join(save_path, "args.txt")}', 'w') as f:
        if hasattr(args, "func"):
            del args.func
        json.dump(args.__dict__, f, indent=2)


def train_function(args):
    ray.init()
    env, env_config = select_env(args)

    if args.algorithm == "ppo":
        model_config = PPOConfig()
    elif args.algorithm == "sac":
        model_config = SACConfig()
    else:
        raise ValueError("Algorithm not supported")
    num_gpus = 1 if torch.cuda.is_available() else 0
    config = (
        model_config
        .environment(env=env, env_config=env_config, normalize_actions=False, clip_actions=True)
        .debugging(seed=args.seed)
        .evaluation(evaluation_interval=1,
                    evaluation_duration=50,
                    evaluation_parallel_to_training=True,
                    evaluation_config=model_config.overrides(explore=False),
                    evaluation_num_workers=2)
        .resources(num_gpus=num_gpus)
        .callbacks(MyCallbacks)
        .framework("torch")
    )
    # create name for train
    if args.comment is None:
        comment = input("Please enter a comment for this training: ")
        args.comment = comment
    name = args.env_model + "_" + args.algorithm + "_" + str(args.seed) + "_" + str(args.comment)

    algo = config.build(logger_creator=custom_log_creator('/mnt/host/mnt/hdd/rllib_prefvec', name))
    save_training_args(args, algo.logdir)

    # training the agent
    train_algo(algo, exit_threshold=args.exit_threshold, max_iterations=args.max_iterations, verbose=True)


if __name__ == "__main__":
    # Create argparse for different options
    main_parser = argparse.ArgumentParser(description='PrefVeC trainer.', add_help=False)
    # Creating subparser element for different execution in case of different arguments
    subparser = main_parser.add_subparsers(required=False)
    # Constructing the env parser
    env_parser = argparse.ArgumentParser(description='Environment parser', add_help=False)
    env_parser.add_argument("--env_model", type=str, default="intersection",
                            choices=["intersection", "grid"])
    env_parser.add_argument("--use_step_reward", type=str2bool, default=False)
    env_parser.add_argument("--default_w", type=float, nargs='+', action=store_as_array, default=[-1000, 1, 1],
                            help="If None, all the objects will get reward 1. "
                                 "Note: this should be 1 for all, and the model wrapper's w should be used.")
    env_parser.add_argument("--env_seed", type=int, default=None)
    env_parser.add_argument("--go_straight", type=str2bool, default=False)
    env_parser.add_argument("--init_vehicles", type=int, default=10)
    env_parser.add_argument("--spawn_prob", type=float, default=0.6)
    env_parser.add_argument("--sim_freq", type=int, default=15)
    env_parser.add_argument("--policy_freq", type=int, default=1)

    trainer_parser = subparser.add_parser("train", description='Trainer parser',
                                          parents=[env_parser], add_help=False,
                                          conflict_handler='resolve',
                                          aliases=['t'])
    trainer_parser.add_argument("--algorithm", type=str, default="ppo",
                                choices=["ppo", "sac"])
    trainer_parser.add_argument('--seed', type=int, default=42, help='Random seed for all numpy/torch operations.')
    trainer_parser.add_argument("--exit_threshold", type=float, default=10.,
                                help="The exit threshold for the training.")
    trainer_parser.add_argument("--max_iterations", type=int, default=90000,
                                help="The maximum number of episodes for the training.")
    trainer_parser.add_argument("--comment", type=str, default=None,
                                help="Sets the comment of the training."
                                     "If None, it will ask for a comment before beginning training")

    # Adding default run mode, especially it is for the pycharm execution.
    if len(sys.argv) == 1:
        sys.argv.insert(1, 'train')
    # Parsing the args from the command line
    args = main_parser.parse_args()
    # Running desired script based on the input.
    train_function(args)
