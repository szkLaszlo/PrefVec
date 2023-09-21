"""
@author "Laszlo Szoke" <szoke.laszlo@kjk.bme.hu>
This file contains script to run the PrefVeC trainings.
"""
import argparse
import sys

from numpy.distutils.fcompiler import str2bool

from PrefVeC.train_test_compose import model_train, model_batch_test, model_test
from PrefVeC.utils.utils import store_as_array, str2dict, store_as_array2

if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                        __        __   __   ___  __    #
    #   |\/|  /\  | |\ |    |__)  /\  |__) /__` |__  |__)   #
    #   |  | /~~\ | | \|    |    /~~\ |  \ .__/ |___ |  \   #
    #                                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # the subparsers are created from this
    main_parser = argparse.ArgumentParser(description='PrefVeC trainer.', add_help=False)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Creating subparser element for different execution in case of different arguments
    subparser = main_parser.add_subparsers(required=False)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Constructing the env parser
    env_parser = argparse.ArgumentParser(description='Environment parser', add_help=False)
    env_parser.add_argument("--env_model", type=str, default="merge",
                            choices=["merge", "intersection", "grid"])
    env_parser.add_argument("--grid_type", type=str, default="semi_static",
                            choices=["random", "static", "semi_static"], )
    env_parser.add_argument("--obs_type", type=str, default="grid", choices=["grid"])
    env_parser.add_argument("--use_step_reward", type=str2bool, default=False)
    env_parser.add_argument("--arena_size", type=int, default=10,
                            help="Size of the arena we want to use")
    env_parser.add_argument("--num_object_types", type=int, default=None,
                            help="Defines how many different object we want.")
    env_parser.add_argument("--max_num_steps", type=int, default=15,
                            help="Defines the max step the agent can take in an episode")
    env_parser.add_argument("--num_init_objects", type=int, default=10,
                            help="Defines how many objects to have at the beginning of each episode.")
    env_parser.add_argument("--default_w", type=float, nargs='+', action=store_as_array2, default=None,
    # todo: use this line is for Q-agent learning
    # env_parser.add_argument("--default_w", type=float, nargs='+', action=store_as_array, default=[-1000, 1],
                            help="If None, all the objects will get reward 1. "
                                 "Note: this should be 1 for all, and the model wrapper's w should be used.")
    # SUMO related parameters.
    env_parser.add_argument("--simulation_directory", type=str, default='./sumo_simulations',
                            help="This is where the simulations are loaded from.")
    env_parser.add_argument("--type_os", type=str, default="merge",
                            help="The observation space type. It can be image or structured")
    env_parser.add_argument("--type_as", type=str, default="discrete_longitudinal",
                            help="The action space type. It can be discrete or continuous")
    env_parser.add_argument("--reward_type", type=str, default="merge",
                            help="Defines how the rewarding is done. See the environment")
    env_parser.add_argument("--use_random_speed", type=str2bool, default=True,
                            help="Defines if we want to use random speed for the ego vehicle")
    env_parser.add_argument("--mode", type=str, default="none",
                            help="Defines if we want to render the environment. Can be none or human")
    env_parser.add_argument("--change_speed_interval", type=int, default=None,
                            help="Defines how often to change the desired speed of the ego")
    env_parser.add_argument("--save_log_path", type=str, default=None,
                            help="Defines where to save the simulation data., if None, a timestamp will be used.")
    env_parser.add_argument("--env_seed", type=int, default=None)
    env_parser.add_argument("--go_straight", type=str2bool, default=False)
    env_parser.add_argument("--vehicles_to_see", type=int, default=15)
    env_parser.add_argument("--init_vehicles", type=int, default=10)
    env_parser.add_argument("--spawn_prob", type=float, default=0.6)
    env_parser.add_argument("--sim_freq", type=int, default=15)
    env_parser.add_argument("--policy_freq", type=int, default=1)
    env_parser.add_argument("--radar_range", type=list, default=[100, 50],
                            help="Sets the radar range for x and y direction")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    model_parser = argparse.ArgumentParser(description='Model parser', add_help=False)

    model_parser.add_argument("--model_version", type=str, default="cl", choices=["q", "cl", ])
    model_parser.add_argument("--model_train_type", type=str, default="sequential",
                              choices=["parallel", "q", "sequential", "dynamic", "contiQ"])
    model_parser.add_argument("--preference_type", type=str, default="cl", choices=["cl", "default"])
    model_parser.add_argument("--network_type", type=str, default="mlp", choices=["mlp", "attention"])
    model_parser.add_argument("--use_double_model", type=str2bool, default=True)
    model_parser.add_argument("--weight_loss_with_sf", type=str2bool, default=True)
    model_parser.add_argument("--copy_policy", type=str2bool, default=True,
                              help="If true, sequential training will initiate with the previous policy weights")
    model_parser.add_argument("--model_hidden_size", type=int, default=64)
    model_parser.add_argument("--last_layer_activation", type=str, default="sigmoid")

    wrapper_parser = argparse.ArgumentParser(description='Model parser', add_help=False)
    wrapper_parser.add_argument("--w", type=float, nargs='+', action=store_as_array,
                                default=[
                                    [0, 1],
                                    [-2, 1],
                                    [-50, 1],
                                    [-100, 1],
                                    [-500, 1],
                                    [-1000, 1],
                                ],
                                help="Successor feature weights. Note: when using from terminal, "
                                     "you have to start with the number of policies, "
                                     "and you need to list the elements one by one")
    wrapper_parser.add_argument("--eval_w", type=float, nargs='+', action=store_as_array, default=[[-1000, 1]])
    wrapper_parser.add_argument("--mix_policies", type=str2bool, default=False,
                                help="If set to true all policies are informed of the others mistakes.")

    wrapper_parser.add_argument("--memory_type", type=str, default="PER", choices=["PER", "replay"])
    wrapper_parser.add_argument("--mem_alpha", type=float, default=0.9)
    wrapper_parser.add_argument("--mem_beta", type=float, default=0.6)
    wrapper_parser.add_argument("--replay_memory_size", type=float, default=500000,
                                help="Defines the size of the buffer")
    wrapper_parser.add_argument("--save_replay_memory", type=str2bool, default=False)
    wrapper_parser.add_argument("--gamma", type=float, default=0.99,
                                help="Defines the discount factor for the training")
    wrapper_parser.add_argument("--update_target_after", type=int, default=100,
                                help="Defines after how many updates to update the target network")

    trainer_parser = subparser.add_parser("train", description='Trainer parser',
                                          parents=[env_parser, model_parser, wrapper_parser], add_help=False,
                                          conflict_handler='resolve',
                                          aliases=['t'])

    trainer_parser.add_argument("--optimizer", default='Adam')
    trainer_parser.add_argument("--learning_rate", type=float, default=0.001)
    trainer_parser.add_argument("--weight_decay", default=0.0, type=float,
                                help="Defines the coefficient (weight) of the used weight decay regularization.")
    trainer_parser.add_argument("--lr_scheduler", type=str2dict, default=None)
    trainer_parser.add_argument("--average_after", type=int, default=100,
                                help="How many episodes to average in logging")
    trainer_parser.add_argument("--observed_time_steps", type=int, default=1,
                                help="Defines how many time steps to give to the network")
    trainer_parser.add_argument("--use_gpu", default=False, type=str2bool,
                                help="If true we use the gpu")
    trainer_parser.add_argument("--batch_size", type=int, default=128)
    trainer_parser.add_argument("--batch_update_num", type=int, default=10,
                                help="How many times to update the network at once")
    trainer_parser.add_argument("--eps_decay", type=float, default=40,
                                help="Defines the percentage of the max_episode where the decay ends")
    trainer_parser.add_argument("--eps_start", type=float, default=0.5,
                                help="Defines the epsilon decay start value.")
    trainer_parser.add_argument("--eps_stop", type=float, default=0.01,
                                help="Defines the minimum value of the eps decay.")
    trainer_parser.add_argument("--eps_reset_rate", type=float, default=None,
                                help="Defines the period in episodes we reset the epsilon decay.")
    trainer_parser.add_argument("--evaluate_after", type=int, default=500,
                                help="After how many episodes to evaluate the model")
    trainer_parser.add_argument("--log_after", type=int, default=50,
                                help="After how many episodes to log the model")
    trainer_parser.add_argument("--evaluate_num_episodes", type=int, default=100,
                                help="After how many episodes to evaluate the model")

    trainer_parser.add_argument("--use_tensorboard", type=str2bool, default=True)
    trainer_parser.add_argument("--log_env", type=str2bool, default=False)
    trainer_parser.add_argument("--continue_training", type=str, default=None,
                                help="If not None, the weights are loaded from the given path, "
                                     "and the training is continued.")
    trainer_parser.add_argument("--save_path", type=str, default=None,
                                help="If None then it will be created based on the time.")
    trainer_parser.add_argument("--render_video_freq", type=int, default=None,
                                help="Defines after how many episodes we want to take videos.")
    trainer_parser.add_argument("--max_episodes", type=int, default=90000,
                                help="Defines how many episodes to run.")
    trainer_parser.add_argument("--seed", type=int, default=5)
    trainer_parser.add_argument("--comment", type=str, default=None,
                                help="Sets the comment of the training."
                                     "If None, it will ask for a comment before beginning training")

    trainer_parser.set_defaults(func=model_train)

    eval_parser = subparser.add_parser("eval", description='Eval parser',
                                       parents=[], add_help=False,
                                       conflict_handler='resolve',
                                       aliases=['e'])
    eval_parser.add_argument("--eval_w", type=float, nargs='+', default=[[-1000.0, 1.0], ])
    eval_parser.add_argument("--env_seed", type=int, default=None)
    eval_parser.add_argument("--simulation_directory", type=str, default='./sumo_simulations',
                             help="This is where the simulations are loaded from.")
    eval_parser.add_argument("--seed", type=int, default=None)
    eval_parser.add_argument("--max_episodes", type=int, default=1000,
                             help="Defines how many episodes to run.")
    eval_parser.add_argument("--render_video_freq", type=int, default=100,
                             help="Defines after how many episodes we want to take videos.")
    eval_parser.add_argument("--mode", default="human")
    eval_parser.add_argument("--use_tensorboard", type=str2bool, default=True)
    eval_parser.add_argument("--log_env", type=str2bool, default=True)
    eval_parser.add_argument("--load_path", type=str, help="Must contain a .weight file",
                             default="/cache/cl/your/model/path")
    eval_parser.add_argument("--use_gpu", default=False, type=str2bool,
                             help="If true we use the gpu")
    eval_parser.add_argument("--is_evaluation", default=True, type=str2bool)
    eval_parser.add_argument("--radar_range", type=list, default=[150, 50],
                             help="Sets the radar range for x and y direction")
    eval_parser.set_defaults(func=model_test)

    batch_eval_parser = subparser.add_parser("batch_eval", description='Batch eval parser',
                                             parents=[eval_parser], add_help=False,
                                             conflict_handler='resolve',
                                             aliases=['b'])

    batch_eval_parser.set_defaults(func=model_batch_test)
    # Adding default run mode, especially it is for the pycharm execution.
    if len(sys.argv) == 1:
        sys.argv.insert(1, 'train')
    # Parsing the args from the command line
    args = main_parser.parse_args()
    # Running desired script based on the input.
    args.func(args)
