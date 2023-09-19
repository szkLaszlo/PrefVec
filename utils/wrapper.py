"""
This file contains a trainer and evaluation wrapper for models.
"""
import copy
import itertools
import os
import pickle
import time
import warnings

import numpy as np
import torch
import tqdm
from torch import optim

from PrefVeC.utils.helper_functions import search_similar_key_in_dict
from PrefVeC.utils.utils import select_optimizer, manage_global_stats, update_log_dict, \
    calculate_episode_reward, Logger

GLOBAL_LOG_PATH = "/cache/scaled_rewards/"


class ModelWrapperInterface(object):

    def __init__(self):
        super(ModelWrapperInterface, self).__init__()
        self.model = NotImplementedError

    def one_episode(self, is_evaluation, log_path):
        """
        This function runs one episode in the environment and returns the episode results.
        :return: dict of the parameters for tensorboard log.
        """
        return NotImplementedError("ModelWrapper's one_episode function is not implemented")

    def update(self, optimizer, memory, batch_size, **kwargs):
        """
        This function runs one model update.
        :return: dict of the parameters for tensorboard log.
        """
        return NotImplementedError("ModelWrapper's update_model function is not implemented")

    def get_trainable_parameters(self):
        return NotImplementedError("ModelWrapper's get_trainable_parameters function is not implemented")

    def update_optimizer(self):
        return NotImplementedError("ModelWrapper's update_optimizer function is not implemented")


class ModelTrainer:

    def __init__(self,
                 model_wrapper,
                 env,
                 memory,
                 train_name,
                 device,
                 batch_size,
                 batch_update_num=10,
                 timesteps_observed=1,
                 eps_start=0.1,
                 eps_decay=0,
                 eps_stop=0.1,
                 optimizer="adam",
                 learning_rate=0.0001,
                 weight_decay=0.001,
                 scheduler=None,
                 average_after=50,
                 log_after=500,
                 early_stopping=False,
                 use_tensorboard=True,
                 continue_training=None,
                 render_video_freq=None,
                 save_path=None,
                 save_memory=False,
                 log_env=False,
                 evaluate_after=None,
                 evaluate_num_episodes=10,
                 is_evaluation=False,
                 eps_reset_rate=None
                 ):

        self.name = train_name
        self.model_wrapper = model_wrapper
        self.env = env
        self.memory = memory
        self.is_evaluation = is_evaluation
        self.batch_size = batch_size
        self.batch_update_num = batch_update_num
        self.device = device
        self.calculate_saliency = is_evaluation and env.rendering
        self.timesteps_observed = timesteps_observed
        self.eps_decay = eps_decay
        self.eps_start = eps_start
        self.eps_stop = eps_stop
        self.eps_reset_rate = eps_reset_rate
        self.render_video_freq = render_video_freq if render_video_freq is not None else evaluate_after
        self.average_after = average_after
        self.log_after = log_after
        self.evaluate_after = evaluate_after
        self.evaluate_num_episodes = evaluate_num_episodes
        self.early_stopping = early_stopping
        # count_parameters(self.model_wrapper.model)
        # Saving path creation if needed
        self.global_training_steps = 0
        self.global_episodes = 0
        self.save_path = save_path \
            if save_path is not None \
            else os.path.join(GLOBAL_LOG_PATH,
                              f'{self.name}/{time.strftime("%Y%m%d_%H%M%S", time.gmtime())}')

        if not os.path.exists(self.save_path) and (use_tensorboard or render_video_freq is not None):
            os.makedirs(self.save_path)
        self.logger = Logger(self.save_path) if use_tensorboard else None
        self.logger.log_graph(self.model_wrapper.model,
                              torch.ones((self.env.observation_space.n,))) if use_tensorboard else None
        setattr(self.memory, "save_path", self.save_path + "/memory_data/") if save_memory else None
        self.log_env = log_env
        if continue_training is not None:
            self.load_weights(continue_training, device=device, req_grad=True)

        self.optimizer = select_optimizer(parameters=self.model_wrapper.get_trainable_parameters(),
                                          optimizer_=optimizer,
                                          learning_rate=learning_rate,
                                          weight_decay=weight_decay)
        self.optimizer_type = optimizer
        self.scheduler = scheduler
        if self.scheduler is not None:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler)

    def train(self, max_episodes):
        """
        This function starts the training.
        :param max_episodes: How many episodes to do.
        :return: None
        """

        # Initiating global variables
        global_stat = manage_global_stats()
        max_eval_reward = -np.inf
        max_eval_success = 0
        stopping_counter = 0
        self.model_wrapper.train()
        trange_ = tqdm.trange(max_episodes, position=0, desc='Training:', leave=True)
        # Running the episodes
        for episode in trange_:
            tb_info = self.run_one_episode(episode)
            trange_.set_description(f"Training episode {episode}: {tb_info['ep_text']}", refresh=True)
            tb_info.pop("ep_text", None)
            self.apply_batch_update(num_updates=self.batch_update_num)
            global_stat = manage_global_stats(stat=global_stat, ep_info=tb_info)
            assert self.global_episodes == episode + 1, "Global episode number is not correct"
            if self.model_wrapper.update_optimizer():
                self.optimizer = select_optimizer(parameters=self.model_wrapper.get_trainable_parameters(),
                                                  optimizer_=self.optimizer_type,
                                                  learning_rate=self.optimizer.defaults["lr"],
                                                  weight_decay=self.optimizer.defaults["weight_decay"])
            if episode % self.log_after == 0:
                self.logger.log_scalar_dict(log_data=tb_info, step=episode) if self.logger is not None else None

            # Calculating average based on a bunch of episodes
            if episode % self.average_after == 0 and episode != 0:
                global_stat.update({"collisions_in_buffer": self.memory.collisions_in_buffer() * self.average_after})
                self.logger.log_scalar_dict(log_data=global_stat, step=self.global_training_steps, prefix="training",
                                            normalizer=self.average_after) if self.logger is not None else None

                if self.scheduler is not None:
                    self.scheduler.step(global_stat.get("loss", 0) / self.average_after)
                    lr = 0
                    for param_group in self.optimizer.param_groups:
                        lr = param_group['lr']
                    self.logger.log_scalar_dict(log_data={"learning_rate": lr}, step=self.global_training_steps,
                                                prefix="training") if self.logger is not None else None

                global_stat = manage_global_stats()

                if stopping_counter > max_episodes * 0.01 and self.early_stopping:
                    print(f"The rewards did not improve since {self.average_after * (stopping_counter - 1)} steps")
                    self.env.stop()
                    break

                else:
                    stopping_counter += 1

            if self.evaluate_after is not None and episode % self.evaluate_after == 0:
                avg_reward, avg_success = self.evaluate(episodes=self.evaluate_num_episodes, render_video=2)
                if avg_reward >= max_eval_reward:
                    self.save_best_weights(episode=episode, suffix="reward")
                    max_eval_reward = avg_reward
                elif avg_success >= max_eval_success:
                    self.save_best_weights(episode=episode, suffix="success")
                    max_eval_success = avg_success
                else:
                    self.save_best_weights(episode=episode, suffix="general")

        avg_reward, avg_success = self.evaluate(episodes=1000, render_video=2)
        print(
            f"Final evaluation after {self.global_training_steps}: Average reward: {avg_reward}, success: {avg_success}")
        # Saving final weights
        torch.save(self.model_wrapper.model.state_dict(), os.path.join(self.save_path, 'model_final.weight'))

    def apply_batch_update(self, num_updates, is_logging=True):
        # Updating the network
        if len(self.memory) > self.batch_size * num_updates:
            # warn user that this is just for grid env
            warnings.warn("A max number of model update/episode  is used. This is just for grid env!!", FutureWarning, )
            updates_to_perform = min(
                self.global_episodes - self.model_wrapper.current_update // self.batch_update_num + num_updates, 20)
            for _ in range(updates_to_perform):
                update_loss = self.model_wrapper.update(optimizer=self.optimizer, memory=self.memory,
                                                        batch_size=self.batch_size)
                self.logger.log_scalar_dict(log_data=update_loss, step=self.model_wrapper.current_update,
                                            prefix="epoch") if self.logger is not None else None

    def run_one_episode(self, episode):
        # Run an episode
        tb_info = self.one_episode(is_evaluation=False,
                                   log_path=f"{self.save_path}/env/{episode}.pkl" if self.log_env else None)
        self.global_episodes += 1
        # Turn off the render
        if self.render_video_freq is not None and (
                episode % self.render_video_freq == 0):
            self.env.save_episode(path=f"{self.save_path}/videos/",
                                  video_name=f"episode_{episode}.avi",
                                  frame_rate=10, scale_percent=1)
        return tb_info

    def save_best_weights(self, episode, suffix="", extention=".weight", delete_previous=True):
        """
        This function saves the current weights.
        It also removes the previously saved ones if delete_previous is True.
        """
        if delete_previous:
            current_weigth_list = os.listdir(self.save_path)
            for item in current_weigth_list:
                if item.endswith(f"{suffix}{extention}"):
                    os.remove(os.path.join(self.save_path, item))
        # Saving weights with better results
        torch.save(self.model_wrapper.model.state_dict(),
                   os.path.join(self.save_path, f'model_{episode + 1}_{suffix}{extention}'))

    def evaluate(self, episodes, path=None, render_video=5):
        """
        This function loads the weights of the trained model, and runs the evaluation.
        """
        if path is not None:
            self.load_weights(path, device=self.device)
        self.model_wrapper.eval()
        eval_stat = manage_global_stats(is_evaluation=True)
        trang_ = tqdm.trange(episodes, position=0, desc="Eval episode", leave=True)
        has_seed = copy.deepcopy(self.env.seed_)
        with torch.no_grad():
            for episode in trang_:
                if not isinstance(has_seed, (int,)):
                    self.env.seed(episode)
                tb_info = self.one_episode(is_evaluation=True,
                                           log_path=f"{self.save_path}/env/{episode}.pkl" if self.log_env else None)

                # self.handle_tensorboard(tb_info, episode)
                trang_.set_description(f"Eval episode {episode}: {tb_info['ep_text']}", refresh=True)
                tb_info.pop("ep_text", None)
                eval_stat = manage_global_stats(eval_stat, tb_info, is_evaluation=True)

                if render_video and episode % self.render_video_freq == 0 or tb_info.get("collision_0", 0):
                    self.env.save_episode(path=f"{self.save_path}/videos/",
                                          video_name=f"eval_{episode}_step{self.global_training_steps}.avi",
                                          frame_rate=10, scale_percent=1)

        self.logger.log_scalar_dict(log_data=eval_stat, step=self.global_episodes, prefix="evaluation",
                                    normalizer=episodes) if self.logger is not None else None

        avg_done, avg_reward = self.print_evaluation_statistics(episodes, eval_stat)
        if not isinstance(has_seed, (int,)):
            self.env.seed(has_seed)
        self.model_wrapper.train()

        return avg_reward, avg_done

    def print_evaluation_statistics(self, episodes, eval_stat):
        avg_reward = search_similar_key_in_dict(eval_stat, 'reward') / episodes
        avg_steps = search_similar_key_in_dict(eval_stat, 'steps') / episodes
        avg_done = search_similar_key_in_dict(eval_stat, 'success', default=0) / episodes
        avg_collision = search_similar_key_in_dict(eval_stat, 'collision', default=0) / episodes
        print(
            f"Evaluation after {self.global_training_steps} steps \t rewards: {avg_reward}," f" average steps: {avg_steps},"
            f" success rate: {avg_done}, collisions: {avg_collision}")

        return avg_done, avg_reward

    def load_weights(self, path_, device, req_grad=False):
        """
        This function loads the defined weights from the path_ and loads it to the model.
        :param: path_: the concrete path of the model weights.
        :return: None
        """
        state_dicts = torch.load(path_, map_location=device)
        print(self.model_wrapper.model.load_state_dict(state_dicts), f"from {path_}")
        for name_, param in self.model_wrapper.model.named_parameters():
            if param.requires_grad:
                param.requires_grad = req_grad
            if "sf" in name_:
                param.requires_grad = False

    def one_episode(self, is_evaluation=False, log_path=None):
        """
        This function runs one episode and returns the information about it.
        :return: Dict containing the following data:
        {
                'steps': t + 1,
                'reward': episode_reward,
                'success': 1 if info['cause'] is None else 0,
                'speed': sum(running_speed) / len(running_speed),
                'lane_change': info['lane_change'],
                'distance': info["distance"],
                'type': self.env.simulation_list.index(self.env.sumoCmd[2]),
        }

        """
        # Init log values
        done, error_running_traci = False, False

        state_list = []
        action_list = []
        reward_list = []

        slm_list = []
        acting_policy_log_data = []
        info_dict = {}
        ep_text = ""
        info = {}
        t = 0
        episode_reward = 0

        # Selecting the policy to train with
        if not is_evaluation:
            self.model_wrapper.set_random_preference_index()

        current_preference = self.model_wrapper.get_preference_index(is_evaluation=is_evaluation)
        current_step = self.global_episodes % self.eps_reset_rate if self.eps_reset_rate is not None else self.global_episodes
        r = max(0.0, 1.0 - current_step / self.eps_decay)
        eps_threshold = self.eps_stop + (self.eps_start - self.eps_stop) * r

        # this while is needed due to unhandled sumo errors
        while not done and not error_running_traci:
            # resetting env
            state_list = [self.env.reset()]
            action_list = []
            acting_policy_log_data = []
            reward_list = []
            slm_list = []
            t = 0
            error_running_traci = False
            # going through the episode step-by-step
            for t in itertools.count():
                # creating the state for the input. Usually 1 timestep but for sumo it may be more.
                current_state = np.expand_dims(state_list[-1], axis=0)
                sample = np.random.random((1,))
                # Selecting action based on current state
                action, acting_policy = self.model_wrapper.forward(current_state, j=current_preference,
                                                                   best_action=eps_threshold < sample or is_evaluation,
                                                                   best_policy=is_evaluation)
                if acting_policy is not None:
                    if acting_policy.ndim > 1:
                        acting_policy_log_data.append(int(acting_policy[:, action]))
                    else:
                        acting_policy_log_data.append(int(acting_policy))
                    # self.env.display_text_on_gui(name="policy", text=acting_policy_log_data[-1], loc=(0, -7))
                if self.calculate_saliency:
                    slm = self.model_wrapper.get_saliency_map(current_state, j=current_preference,
                                                              best_policy=is_evaluation)
                    display_text = str([f"{i:1.4f}" for i in slm])
                    # todo: make this work with sumo.
                    self.env.display_text_on_gui(name="firings", text=display_text, loc=(0, 0))
                    self.env.display_text_on_gui(name="state", text=current_state.squeeze().tolist(), loc=(350, 0))
                    self.env.display_text_on_gui(name="action", text=[action], loc=(0, 350))
                    if acting_policy is not None and acting_policy.ndim > 1:
                        self.env.display_text_on_gui(name="policies", text=list(acting_policy.squeeze()), loc=(0, 400))
                    slm_list.append(slm)

                # Step through environment using chosen action
                # try catch is due to the unhandled sumo errors.

                try:
                    next_state, reward, done, info = self.env.step(action.item())
                    # next_state[5] + next_state[10] + next_state[15] + next_state[20] + next_state[25] == len(self.env.road.vehicles)-1
                    info_dict = update_log_dict(dict=info_dict, key="info", value=info)
                    reward_ = info.get("cumulants", reward) if self.model_wrapper.use_cumulants else reward

                except RuntimeError as err:
                    self.env.reset()
                    reward_list = []
                    error_running_traci = True
                    break

                # adding state to the history
                state_list.append(copy.deepcopy(next_state))
                action_list.append(action.item())

                if not is_evaluation:
                    trans = self.model_wrapper.create_transition(state=state_list[-2].tolist(), action=action_list[-1],
                                                                 next_state=state_list[
                                                                     -1].tolist() if not done else float('nan'),
                                                                 reward=reward_,
                                                                 index=current_preference)

                    self.memory.push(trans, is_rare=True if "collision" == info.get("cause", None) else False)

                # Save reward
                reward_list.append(reward_)

                # Printing to console if episode is terminated
                if done:
                    # assert next_state is None, "next state should be None when done"
                    pref_vector = self.model_wrapper.get_preference_vector(index=current_preference,
                                                                           is_evaluation=is_evaluation)
                    if pref_vector is None:
                        dummy_w = getattr(self.env, "default_w", None)
                        pref_vector = torch.tensor(dummy_w,
                                                   dtype=torch.float32) if dummy_w is not None and self.model_wrapper.use_cumulants else None
                    else:
                        pref_vector = pref_vector.cpu()
                    episode_reward = calculate_episode_reward(reward=reward_list,
                                                              w=pref_vector)

                    self.global_training_steps += t + 1 if not is_evaluation else 0
                    ep_text = f"Steps:{t + 1}, reward: {episode_reward:.3f}, " \
                              f"cause: {info.get('cause', None)}, " \
                              f"policy:{list(pref_vector.numpy()) if pref_vector is not None else 'default'}, " \
                              f"distance:{info.get('distance', 0):.2f}"
                    break

        # logging for the env evaluation
        if log_path is not None:
            if not os.path.exists(os.path.dirname(log_path)):
                os.makedirs(os.path.dirname(log_path))
            with open(log_path, "bw") as file:
                log_dict = {"state": [s.tolist() for s in state_list[:-1]],
                            "action": action_list,
                            "reward": reward_list,
                            "cause": info.get('cause', "nan"),
                            "slm": slm_list,
                            "acting_policy": acting_policy_log_data}
                pickle.dump(log_dict, file)
        success = 1 if info.get("cause", None) is None else 0
        slow = 1 if str(info.get("cause", "None")) in "slow" else 0
        collision = 1 if str(info.get("cause", "None")) in "collision" else 0
        # Used for tensorboard visualisation. Must contain steps, reward, success
        if is_evaluation:
            current_preference = 0
        episode_info = {
            f'steps_{current_preference}': t + 1,
            f'reward_{current_preference}': episode_reward,
            f'slow_{current_preference}': slow,
            f'collision_{current_preference}': collision,
            f'success_{current_preference}': success,
            f'distance': info.get("distance", 0),
            f'action_list': action_list,
            f'acting_policy': acting_policy_log_data if len(acting_policy_log_data) > 0 else 0,
            f'ep_text': ep_text,
            'eps': eps_threshold
        }

        return episode_info
