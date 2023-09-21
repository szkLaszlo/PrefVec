"""
@author "Laszlo Szoke" szoke.laszlo@kjk.bme.hu
"""
import copy
import json
import os
import subprocess
import time

import numpy
import torch.cuda

from continuousSUMO.sumoGym.environment import makeContinuousSumoEnv
from PrefVeC.model_prefvec.model import FastRLWrapper, PrefVeC, CLDQNWrapper, Q
from PrefVeC.model_prefvec.preferences import CLPreferenceSelector, DefaultPolicySelector
from PrefVeC.QN.q_agent import DQNWrapper
from PrefVeC.other_envs.grid import GridWorld
from PrefVeC.other_envs.highway_cl import CumulantIntersectionEnv
from PrefVeC.utils.helper_functions import create_naming_convention
from PrefVeC.utils.memory import PrioritizedReplayBuffer, Transition, TransitionFastRL, ReplayMemory
from PrefVeC.utils.networks import SimpleMLP, SimpleGNN
from PrefVeC.utils.utils import find_latest_weight, seed_everything
from PrefVeC.utils.wrapper import ModelTrainer


# from rlagents.rl_agents.agents.common.models import EgoAttentionNetwork


def model_train(args):
    """
    Function to set up training env and model
    :param args: contains all parameters needed for the train
    :return: none
    """
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    if args.comment is None:
        setattr(args, "comment", str(input("------------------------------------------\n"
                                           " Enter a comment for the current training:\n"
                                           "------------------------------------------\n")))
    env, args = select_environment(args)

    model, transition = select_model(args, env)
    modelwrapper = select_train_type(args)
    memory = select_memory_model(args)

    evaluation_preference, w = select_preferences(args, model)

    # creating the model wrapper
    wrapper = modelwrapper(model=model,
                           trans=transition,
                           target_model=copy.deepcopy(model) if args.use_double_model else None,
                           training_w=w,
                           env=env,
                           eval_w=evaluation_preference,
                           gamma=args.gamma,
                           target_after=args.update_target_after,
                           mix_policies=args.mix_policies,
                           device=device,
                           sequential=args.sequential,
                           copy_policy=args.copy_policy,
                           dynamic_preference=args.dynamic_preference,
                           weight_loss_with_sf=getattr(args, "weight_loss_with_sf", True)
                           )

    # Calculate where the decay ends
    decay_ends_at_step = args.eps_decay * args.max_episodes // 100 if args.eps_reset_rate is None else args.eps_decay * args.eps_reset_rate // 100
    name_ = create_naming_convention(args)
    # creating the model trainer
    trainer = ModelTrainer(model_wrapper=wrapper,
                           env=env,
                           memory=memory,
                           train_name=name_,
                           device=device,
                           batch_size=args.batch_size,
                           batch_update_num=args.batch_update_num,
                           timesteps_observed=args.observed_time_steps,
                           eps_start=args.eps_start,
                           eps_decay=decay_ends_at_step,
                           eps_stop=args.eps_stop,
                           optimizer=args.optimizer,
                           learning_rate=args.learning_rate,
                           weight_decay=args.weight_decay,
                           scheduler=args.lr_scheduler,  # None,
                           average_after=args.average_after,
                           use_tensorboard=args.use_tensorboard,
                           continue_training=args.continue_training,
                           save_path=args.save_path,
                           save_memory=args.save_replay_memory,
                           log_env=args.log_env,
                           evaluate_after=args.evaluate_after,
                           render_video_freq=args.render_video_freq,
                           evaluate_num_episodes=args.evaluate_num_episodes,
                           is_evaluation=False,
                           eps_reset_rate=args.eps_reset_rate,
                           log_after=getattr(args, "log_after", 500),
                           )
    # Get the current git commit hash
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    commit_hash = result.stdout.decode().strip()
    setattr(args, "commit_hash", commit_hash)
    # Saving hyperparameters
    with open(f'{trainer.save_path}/args.txt', 'w') as f:
        del args.func
        json.dump(args.__dict__, f, indent=2)

    # training the agent
    trainer.train(max_episodes=args.max_episodes)


def select_preferences(args, model):
    pref_type = getattr(args, "preference_type", "default")

    if "cl" in pref_type:
        w = CLPreferenceSelector(weights=args.w,
                                 scheduler_steps=args.max_episodes // len(args.w) * args.batch_update_num,
                                 average_range=100 * args.batch_update_num,
                                 is_evaluation=getattr(args, 'is_evaluation', False))
        setattr(args, "eps_reset_rate", args.max_episodes // len(args.w))
    else:
        w = DefaultPolicySelector(args.w, is_evaluation=getattr(args, 'is_evaluation', False))

    evaluation_preference = DefaultPolicySelector(args.eval_w, is_evaluation=getattr(args, 'is_evaluation',
                                                                                     False),
                                                  default_policy=None)
    # To eval given policy change to (args.eval_w, <policy index>)

    return evaluation_preference, w


def select_memory_model(args):
    mem_type = getattr(args, "memory_type", "replay")
    # creating the memory replay object
    if mem_type in "PER":
        memory_ = PrioritizedReplayBuffer(buffer_size=args.replay_memory_size, alpha=getattr(args, "mem_alpha", 0.6),
                                          beta=getattr(args, "mem_beta", 0.4))
    elif mem_type in "replay":
        memory_ = ReplayMemory(capacity=args.replay_memory_size)
    else:
        raise NotImplementedError(f"The selected memory type {mem_type} is not implemented")

    return memory_


def select_train_type(args):
    train_type = getattr(args, "model_train_type", "parallel")

    if "sf" in train_type or "parallel" in train_type:
        modelwrapper = FastRLWrapper
        setattr(args, 'sequential', False)
        setattr(args, 'dynamic_preference', False)
        setattr(args, 'copy_policy', False)
    elif "cl" in train_type or "sequential" in train_type:
        modelwrapper = FastRLWrapper
        setattr(args, 'sequential', True)
        setattr(args, 'dynamic_preference', False)
    elif "q" in train_type:
        setattr(args, 'sequential', False)
        setattr(args, 'dynamic_preference', False)
        setattr(args, 'copy_policy', False)
        modelwrapper = DQNWrapper
    elif "dynamic" in train_type:
        setattr(args, 'sequential', True)
        setattr(args, 'dynamic_preference', True)
        setattr(args, 'copy_policy', True)
        modelwrapper = FastRLWrapper
    elif "contiQ" in train_type:
        setattr(args, 'sequential', False)
        setattr(args, 'dynamic_preference', False)
        setattr(args, 'copy_policy', False)
        modelwrapper = CLDQNWrapper
    else:
        raise NotImplementedError

    return modelwrapper


def select_model(args, env):
    nt = getattr(args, "network_type", "mlp")
    network_type = SimpleMLP if nt == "mlp" else SimpleGNN if nt == "graph" else "attention" if nt == "attention" else None

    # Using PrefVeC model
    if args.model_version == "cl":
        # creating composed model
        model = PrefVeC(network_type=network_type, input_size=env.observation_space.n, actions=env.action_space.n,
                        d=args.num_object_types, hidden_size=args.model_hidden_size, num_policies=len(args.w),
                        activation=getattr(args, "last_layer_activation", "sigmoid"))
        transition = TransitionFastRL

    # Using Q-learning model
    elif args.model_version == "q":
        model = Q(network_type=network_type, input_size=env.observation_space.n,
                  actions=env.action_space.n, hidden_size=args.model_hidden_size, activation=None)
        transition = Transition

    else:
        raise RuntimeError("Not implemented model version")

    return model, transition


def select_environment(args):
    if not hasattr(args, 'env_model'):
        setattr(args, 'env_model', 'merge')

    elif args.env_model == "intersection":
        env_configs = {"observation": {
            "type": "Kinematics",
            "vehicles_count": getattr(args, "vehicles_to_see", 6),
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
            },
            "controlled_vehicles": 1,
            "policy_frequency": getattr(args, "policy_freq", 5),
            "initial_vehicle_count": getattr(args, "init_vehicles", 1),
            "simulation_frequency": getattr(args, "sim_freq", 10),
            "spawn_probability": getattr(args, "spawn_prob", 0.1),
            "duration": 13,  # [s]
            'collision_reward': 1, 'high_speed_reward': 1,
            'arrived_reward': 1,
            'reward_speed_range': [7.0, 9.0], 'normalize_reward': False,
            "offscreen_rendering": False if args.mode == "human" else True,
            "video": True if args.mode == "video" else False,
        }
        config = getattr(args, "env_config", env_configs)
        config["offscreen_rendering"] = False if args.mode == "human" else True
        config["video"] = True if args.mode == "video" else False
        env = CumulantIntersectionEnv(default_w=args.default_w,
                                      config=config, go_straight=getattr(args, "go_straight", False),
                                      observed_timesteps=args.observed_time_steps)

        setattr(args, "env_config", env.config)

    elif args.env_model == "merge":

        env = makeContinuousSumoEnv('SUMOEnvironment-v0',
                                    simulation_directory=args.simulation_directory,
                                    type_os=args.type_os,
                                    type_as=args.type_as,
                                    reward_type=args.reward_type,
                                    mode=args.mode,
                                    radar_range=getattr(args, "radar_range", [100, 50]),
                                    save_log_path=None,
                                    change_speed_interval=args.change_speed_interval,
                                    default_w=args.default_w,
                                    seed=getattr(args, 'env_seed', None),
                                    use_random_speed=getattr(args, 'use_random_speed', False),
                                    )

    elif args.env_model == "grid":
        env = GridWorld(env_config={"n": args.arena_size,
                                    "default_w": args.default_w,
                                    "offscreen_rendering": False if args.mode == "human" else True,
                                    "video": True if args.mode == "video" else False,
                                    "grid_type": getattr(args, "grid_type", "static"),
                                    "obs_type": getattr(args, "obs_type", "grid"),
                                    "use_step_reward": getattr(args, "use_step_reward", False),
                                    })
    else:
        raise NotImplementedError

    if args.w is None:
        setattr(args, "w", numpy.eye(env.get_max_reward(1).shape[0]).tolist())

    if args.eval_w is None:
        setattr(args, "eval_w", [numpy.ones(env.get_max_reward(1).shape[0]).tolist()])
        setattr(args, "evaluate_after", None)
    else:
        setattr(args, "evaluate_after", getattr(args, "evaluate_after", 1000))

    if args.num_object_types is None:
        setattr(args, "num_object_types", env.get_max_reward(1).shape[0])

    return env, args


def model_batch_test(args):
    base_dir = args.load_path
    directories_to_test = os.listdir(args.load_path)
    for dir in directories_to_test:
        args1 = copy.deepcopy(args)
        if os.path.isdir(os.path.join(base_dir, dir)):
            setattr(args1, "load_path", os.path.join(base_dir, dir))
            try:
                print(f"Inferencing model in {dir}...")
                model_test(args1)
            except Exception as inst:
                print(f"There was an error during inferencing model in {dir}. \n {inst.args} Please check! Skipping...")


def model_test(args):
    """
    Function to evaluate model
    :param args: contains all parameters for the evaluation
    :return: None
    """
    train_args = copy.deepcopy(args)
    seed_everything(args.seed)
    weight_path = args.load_path
    # Loading saved argparse elements
    if os.path.isfile(args.load_path):
        setattr(args, "load_path", os.path.dirname(args.load_path))
    with open(f'{args.load_path}/args.txt', 'r') as f:
        train_args.__dict__ = json.load(f)
    # replacing train parameters in evaluation args
    for key_, value_ in train_args.__dict__.items():
        if not hasattr(args, key_):
            setattr(args, key_, value_)
    device = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    setattr(args, "eps_reset_rate", None)
    env, args = select_environment(args)

    model, transition = select_model(args, env)
    modelwrapper = select_train_type(args)

    memory = select_memory_model(args)

    evaluation_preference, w = select_preferences(args, model)

    # creating the model wrapper
    wrapper = modelwrapper(model=model,
                           trans=transition,
                           training_w=w,
                           env=env,
                           eval_w=evaluation_preference,
                           gamma=args.gamma,
                           target_after=args.update_target_after,
                           mix_policies=args.mix_policies,
                           device=device,
                           sequential=args.sequential,
                           copy_policy=args.copy_policy,
                           dynamic_preference=args.dynamic_preference
                           )

    # Calculate where the decay ends
    decay_ends_at_step = args.eps_decay * args.max_episodes // 100 if args.eps_reset_rate is None else args.eps_decay * args.eps_reset_rate // 100
    # creating the model trainer
    trainer = ModelTrainer(model_wrapper=wrapper,
                           env=env,
                           memory=memory,
                           train_name="eval",
                           device=device,
                           batch_size=args.batch_size,
                           batch_update_num=10,
                           timesteps_observed=args.observed_time_steps,
                           eps_start=0,
                           eps_decay=1,
                           eps_stop=0,
                           eps_reset_rate=None,
                           optimizer=args.optimizer,
                           learning_rate=args.learning_rate,
                           weight_decay=args.weight_decay,
                           scheduler=None,
                           average_after=10,
                           use_tensorboard=args.use_tensorboard,
                           log_after=1,
                           continue_training=None,
                           save_path=os.path.join(args.load_path,
                                                  f'eval_{time.strftime("%Y%m%d_%H%M%S", time.gmtime())}'),
                           log_env=args.log_env,
                           render_video_freq=args.render_video_freq,
                           evaluate_after=args.evaluate_after,
                           evaluate_num_episodes=args.evaluate_num_episodes,
                           is_evaluation=True,
                           )
    # Saving eval parameters
    with open(f'{trainer.save_path}/args_eval.txt', 'w') as f:
        if hasattr(args, "func"):
            del args.func
        json.dump(args.__dict__, f, indent=2)
    # Finding the last weight from the saved ones
    weigth_path = find_latest_weight(path=weight_path)
    # Evaluate the model
    trainer.evaluate(episodes=args.max_episodes, path=weigth_path, render_video=1)
    env.stop()
