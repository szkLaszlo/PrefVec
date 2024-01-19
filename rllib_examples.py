"""
This file contains code for running our environments with rllib.
"""
import os
import tempfile
from datetime import datetime
from typing import Dict

import ray
from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.evaluation import Episode

from PrefVeC.other_envs.grid import GridWorld
from PrefVeC.other_envs.highway_cl import CumulantIntersectionEnv
from RobustRL.utils.utils import train_algo


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
            cause_list = ["standing_still", "success", "far", "collision", "slow", "wrong_direction"]
            for cause_i in cause_list:
                if cause_i in cause:
                    episode.custom_metrics[f"{cause_i}"] = 1
                else:
                    episode.custom_metrics[f"{cause_i}"] = 0
            travelled_dist = info['travelled_dist']
            episode.custom_metrics["travelled_dist"] = travelled_dist
            dist_from_goal = info['dist_from_goal']
            episode.custom_metrics["dist_from_goal"] = dist_from_goal
            angle_to_goal = info['angle_to_goal']
            episode.custom_metrics["angle_to_goal"] = angle_to_goal
        else:
            pass


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


def select_env(env_name):
    if env_name == "grid":
        n = 10
        env = GridWorld
        env_config = {
            "n": n,
            "default_w": [-1000, 1],
            "offscreen_rendering": True,
            "video": True,
            "grid_type": "semi_static",
            "obs_type": "grid",
            "use_step_reward": True}
        return env, env_config

    elif env_name == "intersection":
        env_configs = {
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
            }, "controlled_vehicles": 1, "policy_frequency": 1,
            "initial_vehicle_count": 3,
            "simulation_frequency": 15,
            "spawn_probability": 0.03, "duration": 13, 'collision_reward': 1,
            'high_speed_reward': 1, 'arrived_reward': 1, 'reward_speed_range': [7.0, 9.0], 'normalize_reward': False,
            "offscreen_rendering": True,
            "video": False,
            "default_w": [-1, 1, 1],
            "go_straight": False
        }

        env = CumulantIntersectionEnv
        return env, env_configs


if __name__ == "__main__":
    ray.init(local_mode=True)
    env_type = "intersection"
    env, env_config = select_env(env_type)
    model_config = PPOConfig()
    config = (
        model_config
        .environment(env=env, env_config=env_config, normalize_actions=False, clip_actions=True)
        .resources(num_gpus=0)
        .debugging(seed=42)
        # .callbacks(MyCallbacks)
        .framework("torch")
    )

    algo = config.build(logger_creator=custom_log_creator('/tmp/ray/ray_results', f'sac_{env_type}'))

    # training the agent
    train_algo(algo, exit_treshold=10, max_iterations=1000, verbose=True)
