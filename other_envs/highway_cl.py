import functools
import os
import warnings
from typing import Tuple, Optional, Callable, Union

import numpy as np
import pygame
from gym import spaces
from highway_env import utils
from highway_env.envs import IntersectionEnv, observation_factory
from highway_env.envs.common.abstract import Observation
from highway_env.envs.common.action import ActionType
from highway_env.utils import Vector
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle

from PrefVeC.utils.helper_functions import distance
from PrefVeC.utils.utils import save_images_to_video

warnings.simplefilter(action='ignore', category=FutureWarning)


def calc_distance(a, b):
    return np.sqrt(sum(((a[0] - b[0]) ** 2, (a[1] - b[1]) ** 2)))


class EgoVehicle(MDPVehicle):
    """A vehicle controlled by the agent."""

    def speed_control(self, target_speed: float) -> float:
        return target_speed

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action.

        - If the action is a speed change, choose speed from the allowed discrete range.
        - Else, forward action to the ControlledVehicle handler.

        :param action: a high-level action
        """
        if action is not None:
            self.target_speed = action if action + self.speed > 0 else min(-self.speed, 0)

        super().act()


class DiscreteMeta(ActionType):
    """
    """

    def __init__(self,
                 env: 'AbstractEnv',
                 target_speeds: Optional[Vector] = None,
                 **kwargs) -> None:
        """
        Create a discrete action space of meta-actions.

        :param env: the environment
        :param longitudinal: include longitudinal actions
        :param lateral: include lateral actions
        :param target_speeds: the list of speeds the vehicle is able to track
        """
        super().__init__(env)
        self.target_speeds = np.array(target_speeds) if target_speeds is not None else MDPVehicle.DEFAULT_TARGET_SPEEDS

    def space(self) -> spaces.Space:
        return spaces.Discrete(3)

    @property
    def vehicle_class(self) -> Callable:
        return functools.partial(EgoVehicle, target_speeds=self.target_speeds)

    def act(self, action: int) -> None:
        self.controlled_vehicle.act(action)

class CumulantIntersectionEnv(IntersectionEnv):
    def __init__(self, default_w=None, config=None, **kwargs):
        self.default_w = np.asarray(default_w) if default_w else np.ones((3,))
        self.observation_space_n = len(config["observation"]["features"])
        self.video_history = [] if config.get("video", False) else None
        self.flatten_obs = config["observation"]["flatten"]
        super(CumulantIntersectionEnv, self).__init__(config=config)
        self.seed_ = 42
        self.seed(self.seed_)
        self.rendering = False if self.config["offscreen_rendering"] else True
        setattr(self.observation_space, 'n', self.observation_space_n)
        setattr(self.action_space, 'n', 3)
        self.dt = 1 / config["policy_frequency"]

    def _get_cumulants(self, vehicle: Vehicle):
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        collision_reward = self.config["collision_reward"] * vehicle.crashed
        high_speed_reward = self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        arrived_reward = self.config["arrived_reward"] if self.has_arrived(vehicle) else 0.
        cumulants = np.array([collision_reward, high_speed_reward, arrived_reward], dtype=np.float32)
        return cumulants

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        cumulants = self._get_cumulants(vehicle)
        reward = sum(self.default_w * cumulants)
        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = super().step(action)
        if self.video_history is not None:
            self.video_history.append(self.render('rgb_array'))
        if self.rendering:
            self.render()
        cumulants = self._get_cumulants(self.vehicle)
        info["cumulants"] = cumulants
        info["cause"] = "slow" if done and not self.vehicle.crashed and not self.has_arrived(self.vehicle) else "collision" if self.vehicle.crashed else None

        return obs, reward, done, info

    def get_max_reward(self, temp_reward):
        return np.array([1, 1, 1])

    def set_render(self, mode, save_path=None):
        pass

    def display_text_on_gui(self, name, text=None, loc=None, rel_to_ego=True):
        if self.viewer is not None:
            font1 = pygame.font.SysFont(None, 20)
            for i in range(0, len(text) + 1, 5):
                img1 = font1.render(str([f"{j:1.3f}" for j in text[i:i + 5]]), True, (0, 0, 0))
                self.viewer.screen.blit(img1, (loc[0], loc[1] + i * 5))
            pygame.display.update()
        else:
            pass

    def stop(self):
        pass

    def save_episode(self, path, video_name="video.avi",
                     frame_rate=10, scale_percent=1):
        if self.video_history is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            img = self.video_history[0]
            width = int(img.shape[1] * scale_percent)
            height = int(img.shape[0] * scale_percent)
            dim = (width, height)
            save_images_to_video(self.video_history, path, video_name, frame_rate, dim)

    def seed(self, seed: int = None):
        if isinstance(seed, (np.random.RandomState,)):
            self.np_random = seed
            return None
        else:
            seed_ = super().seed(seed)
            self.seed_ = self.np_random
        return seed_

    def reset(self) -> Observation:
        obs = super().reset()
        if self.video_history is not None:
            self.video_history = []
        return obs


class CumulantWrapper(IntersectionEnv):
    MID_POINT = (2.0, 0.0)
    END_POINT = (-20., -2.0)

    def __init__(self, default_w=None, config=None, **kwargs):
        self.default_w = np.asarray(default_w) if default_w else np.ones((2,))
        self.max_vehicles_observed = config["observation"]["vehicles_count"] - 1
        self.observed_timesteps = kwargs.get("observed_timesteps", 1)
        self.observation_space_n = (self.max_vehicles_observed + 1)*6 * self.observed_timesteps
        self.go_straight = kwargs.get("go_straight", False)
        self.observation_list = []
        self.video_history = [] if config.get("video", False) else None
        self.generate_obs = self.generate_observation
        self.flatten_obs = config["observation"]["flatten"]
        super(CumulantWrapper, self).__init__(config=config)
        self.seed_ = 42
        self.seed(self.seed_)
        self.last_position = None
        self.last_dist = 0
        self.rendering = False if self.config["offscreen_rendering"] else True
        setattr(self.observation_space, 'n', self.observation_space_n)
        setattr(self.action_space, 'n', 3)
        self.discrete_action_list = [-6, 0, 3]
        self.max_distance = None
        self.dt = 1 / config["policy_frequency"]

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        cumulants = self._get_cumulants(vehicle)
        reward = sum(self.default_w * cumulants)
        return reward

    def _get_cumulants(self, vehicle):
        progress = calc_distance(vehicle.position, self.last_position) / 1000 / self.max_distance
        collision_reward = vehicle.crashed
        cumulants = np.array([collision_reward, progress], dtype=np.float32)
        return cumulants

    def generate_observation(self, flatten=True):
        obs = [[1,
            1. - getattr(self, "dt", 0.0) * self.steps / self.config["duration"],
               self.vehicle.position[0] / abs(self.config["observation"]["features_range"]["x"][0]),
               self.vehicle.position[1] / abs(self.config["observation"]["features_range"]["y"][0]),
               self.vehicle.velocity[0] / abs(self.config["observation"]["features_range"]["vx"][0]),
               self.vehicle.velocity[1] / abs(self.config["observation"]["features_range"]["vy"][0])
               ]]

        sorted_vehicles = sorted(self.road.vehicles, key=lambda v: distance(self.vehicle, v))
        first_n_vehicles = sorted_vehicles[1:self.max_vehicles_observed + 1]
        if len(first_n_vehicles) < self.max_vehicles_observed:
            first_n_vehicles += [None] * (self.max_vehicles_observed - len(first_n_vehicles))
        if self.config["observation"]["order"] in "shuffled":
            np.random.shuffle(first_n_vehicles)
        for vehicle in first_n_vehicles:
            if vehicle is None:
                obs += [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            elif vehicle is not self.vehicle:
                obs += [[1.0,
                         0.,
                        vehicle.position[0] / abs(self.config["observation"]["features_range"]["x"][0]),
                        vehicle.position[1] / abs(self.config["observation"]["features_range"]["y"][0]),
                        vehicle.velocity[0] / abs(self.config["observation"]["features_range"]["vx"][0]),
                        vehicle.velocity[1] / abs(self.config["observation"]["features_range"]["vy"][0]),
                        ]]

        self.observation_list.append(np.asarray(obs))
        while len(self.observation_list) < self.observed_timesteps:
            self.observation_list.insert(0, np.zeros_like(obs))
        if len(self.observation_list) > self.observed_timesteps:
            self.observation_list.pop(0)
        return np.stack(self.observation_list).flatten() if flatten else np.stack(self.observation_list)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        action = self.discrete_action_list[action]
        _, reward, done, info = super().step(action)
        obs = self.generate_obs(flatten=self.flatten_obs)
        obs = obs[0] if self.observed_timesteps == 1 else obs
        if self.video_history is not None:
            self.video_history.append(self.render('rgb_array'))
        if self.rendering:
            self.render()
        cumulants = self._get_cumulants(self.vehicle)
        last_pos_x, last_pos_y = float(self.vehicle.position[0]), float(self.vehicle.position[1])
        self.last_position = (last_pos_x, last_pos_y)
        success = False
        if last_pos_x < -20.:
            done = True
            success = True
            cumulants[1] = 1 - self.last_dist
            reward = 1 - self.last_dist
        self.last_dist += cumulants[1]
        info["cumulants"] = cumulants
        info["distance"] = self.last_dist
        info[
            "cause"] = "slow" if done and not self.vehicle.crashed and not success else "collision" if self.vehicle.crashed else None

        return obs, reward, done, info

    def reset(self) -> Observation:
        _ = super().reset()
        obs = self.generate_obs(flatten=self.flatten_obs)
        return obs[0] if self.observed_timesteps == 1 else obs

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])
        self.vehicle.speed = max(0, self.np_random.randint(-2, 9) + self.np_random.random())
        self.last_dist = 0
        self.last_position = self.controlled_vehicles[0].position
        self.max_distance = (calc_distance(self.MID_POINT, self.END_POINT) + calc_distance(self.MID_POINT,
                                                                                           self.last_position)) / 1000  # [km]
        if self.video_history is not None:
            self.video_history = []

    def save_episode(self, path, video_name="video.avi",
                     frame_rate=10, scale_percent=1):
        if self.video_history is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            img = self.video_history[0]
            width = int(img.shape[1] * scale_percent)
            height = int(img.shape[0] * scale_percent)
            dim = (width, height)
            save_images_to_video(self.video_history, path, video_name, frame_rate, dim)

    def seed(self, seed: int = None):
        if isinstance(seed, (np.random.RandomState,)):
            self.np_random = seed
            return None
        else:
            seed_ = super().seed(seed)
            self.seed_ = self.np_random
        return seed_

    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       position_deviation: float = 1.,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.6,
                       go_straight: bool = False) -> None:
        if self.np_random.rand() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if self.go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(self.road, ("o" + str(route[0]), "ir" + str(route[0]), 0),
                                            longitudinal=longitudinal + 5 + self.np_random.randn() * position_deviation,
                                            speed=8 + self.np_random.randn() * speed_deviation)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        #vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def get_max_reward(self, temp_reward):
        return np.array([1, 1])

    def set_render(self, mode, save_path=None):
        pass

    def display_text_on_gui(self, name, text=None, loc=None, rel_to_ego=True):
        if self.viewer is not None:
            font1 = pygame.font.SysFont(None, 20)
            for i in range(0, len(text) + 1, 5):
                img1 = font1.render(str([f"{j:1.3f}" for j in text[i:i + 5]]), True, (0, 0, 0))
                self.viewer.screen.blit(img1, (loc[0], loc[1] + i * 5))
            pygame.display.update()
        else:
            pass

    def stop(self):
        pass

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        self.observation_type = observation_factory(self, self.config["observation"])
        self.action_type = DiscreteMeta(self, **self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()


if __name__ == "__main__":
    env = CumulantWrapper(config={
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20],
            },
            "absolute": True,
            "flatten": True,
            "observe_intentions": False
        },
        "controlled_vehicles": 1,
        "policy_frequency": 10,
        "offscreen_rendering": True
    })
    while True:
        env.reset()
        done = False
        while not done:
            action = np.random.randint(0, 3)
            obs, reward, done, info = env.step(action)
        print(info)
