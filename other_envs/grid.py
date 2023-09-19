import os
import random
import time
from typing import Tuple, List, Generator

import gym
import numpy as np
import pygame
from gym import spaces
from gym.utils import seeding

from PrefVeC.utils.utils import save_images_to_video

END_POS = [0., 0., 1.]
CURRENT_POS = [0., 1., 0.]
PATH = [1., 0., 0.]
EMPTY = 0
grid1 = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0., 0., 1.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1., 0., 0.],
          [1., 0., 0.], [1., 0., 0.], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0.0, 0.0, 0.0]]]
grid2 = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0., 0., 1.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [1., 0., 0.], [1., 0., 0.], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0.0, 0.0, 0.0]]]


def get_path(n: int, A: Tuple[int, int], B: Tuple[int, int], shortest: bool = False, avoid_walls: bool = True) -> List[
    Tuple[int, int]]:
    """
    Generate a path from A to B in an n x n grid world.

    Args:
        n (int): The size of the grid world.
        A (Tuple[int, int]): The starting cell coordinates as a tuple (x, y).
        B (Tuple[int, int]): The target cell coordinates as a tuple (x, y).
        shortest (bool): If True, generate the shortest path from A to B. If False, generate a non-shortest path. Default is True.
        hooks (bool): If True, allow the path to include hooks (i.e., non-straight segments). If False, only generate a straight path. Default is True.
        avoid_walls (bool): If True, avoid generating a path that crosses walls. If False, allow the path to cross walls. Default is True.

    Returns:
        List[Tuple[int, int]]: A list of cell coordinates representing the path from A to B.
    """

    # Define a helper function to get the valid neighbors of a cell.
    def get_neighbors(x: int, y: int, walls: set) -> List[Tuple[int, int]]:
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        valid_neighbors = []
        for nx, ny in neighbors:
            if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in walls:
                valid_neighbors.append((nx, ny))
        return valid_neighbors

    # Define a helper function to get the Euclidean distance between two cells.
    def distance(x1: int, y1: int, x2: int, y2: int) -> float:
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    # Initialize the path to just the start cell A and the visited set to contain A.
    path = [(A[0], A[1])]
    visited = set(path)

    # Initialize the walls to be an empty set.
    walls = set()

    # Generate a path from A to B.
    while True:
        x, y = path[-1]

        # If we have reached B, return the path.
        if x == B[0] and y == B[1]:
            return path

        # Get the valid neighbors of the current cell.
        neighbors = get_neighbors(x, y, walls)

        # If we are generating the shortest path, choose the neighbor with the shortest Euclidean distance to B.
        if shortest:
            neighbor_dists = [distance(nx, ny, B[0], B[1]) for nx, ny in neighbors]
            min_dist = min(neighbor_dists)
            neighbors = [neighbor for neighbor, dist in zip(neighbors, neighbor_dists) if dist == min_dist]

        # If we are generating a non-shortest path, remove any neighbors that are farther from B than the current cell.
        else:
            neighbor_dists = [distance(nx, ny, B[0], B[1]) for nx, ny in neighbors]
            neighbors = [neighbor for neighbor, dist in zip(neighbors, neighbor_dists) if
                         dist <= distance(x, y, B[0], B[1])]

        # If we are avoiding walls, remove any neighbors that are walls and not the goal cell.
        if avoid_walls:
            neighbors = [neighbor for neighbor in neighbors if neighbor == B or neighbor not in walls]

        # If there are no valid neighbors, backtrack until we find one.
        while path and not neighbors:
            path.pop()
            if path:
                x, y = path[-1]
                neighbors = get_neighbors(x, y, walls)

                # If we are generating the shortest path, choose the neighbor with the shortest Euclidean distance to B.
                if shortest:
                    neighbor_dists = [distance(nx, ny, B[0], B[1]) for nx, ny in neighbors]
                    min_dist = min(neighbor_dists)
                    neighbors = [neighbor for neighbor, dist in zip(neighbors, neighbor_dists) if dist == min_dist]

                # If we are generating a non-shortest path, remove any neighbors that are farther from B than the current cell.
                else:
                    neighbor_dists = [distance(nx, ny, B[0], B[1]) for nx, ny in neighbors]
                    neighbors = [neighbor for neighbor, dist in zip(neighbors, neighbor_dists) if
                                 dist <= distance(x, y, B[0], B[1])]

                # If we are avoiding walls, remove any neighbors that are walls and not the goal cell.
                if avoid_walls:
                    neighbors = [neighbor for neighbor in neighbors if neighbor == B or neighbor not in walls]

        # If there are valid neighbors, choose one at random and add it to the path.
        if neighbors:
            if len(path) > 1:
                # Check if we can add a hook to the path
                last_dx = path[-1][0] - path[-2][0]
                last_dy = path[-1][1] - path[-2][1]
                for nx, ny in neighbors:
                    if nx - x == last_dy and ny - y == -last_dx:
                        path.append((nx, ny))
                        visited.add((nx, ny))
                        break
                    elif nx - x == -last_dy and ny - y == last_dx:
                        path.append((nx, ny))
                        visited.add((nx, ny))
                        break
                else:
                    # No valid hooks, choose a neighbor at random
                    nx, ny = random.choice(neighbors)
                    path.append((nx, ny))
                    visited.add((nx, ny))
            else:
                # Choose a neighbor at random
                nx, ny = random.choice(neighbors)
                path.append((nx, ny))
                visited.add((nx, ny))
        # If there are no valid neighbors and we have backtracked all the way to the start, return None.
        else:
            if len(path) == 1:
                return None

            # Otherwise, remove the current cell from the path and add it to the walls set.
            path.pop()
            x, y = path[-1]
            walls.add((x, y))


class GridWorld(gym.Env):
    def __init__(self, env_config, **kwargs):
        self.n = env_config.get("n", 10)
        self.grid = np.zeros((self.n, self.n, 3))
        self.start_pos = (self.n - 1, self.n - 1)
        self.end_pos = (0, 0)
        self.path = []
        self.current_pos = (0, 0)
        default_w = env_config.get("default_w", None)
        self.default_w = np.asarray(default_w, dtype=np.float32) if default_w else np.ones((2,), dtype=np.float32)

        self.action_space = spaces.Discrete(5)  # Up, Down, Left, Right, None
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n - 6, self.n), dtype=np.float32)
        self.grid_type = env_config.get("grid_type", "static")  # static, random, semi_static
        self.obs_type = env_config.get("obs_type", "position")  # grid, position
        assert (self.grid_type in ["random"] and self.obs_type not in ["position"]) or self.grid_type in ["static",
                                                                                                          "semi_static"]
        # setattr(self.action_space, 'n', 4)        # Pygame initialization
        self.screen_size = (400, 400)

        self.cell_size = (self.screen_size[0] // self.n, self.screen_size[1] // self.n)
        self.screen = None
        self.rendering = False if env_config.get("offscreen_rendering", False) else True
        self.video_history = [] if env_config.get("video", False) else None
        self.use_step_reward = env_config.get("use_step_reward", False)
        self.current_step = 0
        self.max_steps = 20
        self.seed_ = 42
        self.seed(self.seed_)
        self.max_dist_to_goal = None
        self.dist_to_goal = None
        setattr(self.observation_space, 'n', self._create_obs().shape[0])

    def reset(self, seed=None, **kwargs):
        env_choice = [0, 0] if self.grid_type in "static" else [0, 1]
        self.grid = np.array(
            [grid1, grid2][self.np_random.choice(env_choice)]) if "static" in self.grid_type else np.zeros(
            (self.n, self.n, 3))
        self.start_pos = tuple(np.concatenate(np.where(self.grid[:, :, 1] == 1.))) if "static" in self.grid_type else (
            self.np_random.randint(0, self.n), self.np_random.randint(0, self.n))

        self.end_pos = tuple(np.concatenate(np.where(self.grid[:, :, 2] == 1.))) if "static" in self.grid_type else (
            self.np_random.randint(0, self.n), self.np_random.randint(0, self.n))
        while "static" not in self.grid_type and self._calc_dist_betw_points(self.start_pos, self.end_pos) < \
                self.grid.shape[0] // 2:
            self.end_pos = (self.np_random.randint(0, self.n), self.np_random.randint(0, self.n))

        # Generate a random path from start to end
        self.path = [tuple(a) for a in
                     np.stack(np.where(self.grid[:, :, 0] == 1.), axis=-1)] if "static" in self.grid_type else get_path(
            self.n,
            self.start_pos,
            self.end_pos)
        # adding the start and end positions to the path
        self.path.append(self.start_pos) if self.start_pos not in self.path else None
        self.path.append(self.end_pos) if self.end_pos not in self.path else None

        # select random start position on path
        if self.np_random.uniform() > 0.5:
            self.current_pos = self.start_pos
        else:
            point = self.path[self.np_random.randint(0, len(self.path))]
            while self._calc_dist_betw_points(point, self.end_pos) < 2:
                point = self.path[self.np_random.randint(0, len(self.path))]
            self.current_pos = point
        self.start_pos = self.current_pos
        # add start position's neighbours to path
        if "static" not in self.grid_type:
            self.path.extend([(min(self.start_pos[0] + 1, self.n - 1), self.start_pos[1]),
                              (max(self.start_pos[0] - 1, 0), self.start_pos[1]),
                              (self.start_pos[0], min(self.start_pos[1] + 1, self.n - 1)),
                              (self.start_pos[0], max(self.start_pos[1] - 1, 0))])
            # updating grid with path elements
            for (x, y) in self.path:
                self.grid[x, y] = PATH
            self.grid[self.current_pos] = CURRENT_POS
            self.grid[self.end_pos] = END_POS

        if self.rendering:
            # Pygame initialization
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("GridWorld")
        if self.video_history is not None:
            self.video_history = [self.grid]
        self.current_step = 0
        self.dist_to_goal = self._calc_dist_betw_points(self.end_pos, self.current_pos)
        self.max_dist_to_goal = self.dist_to_goal
        state_ = self._create_obs()
        assert self.end_pos in self.path and self.start_pos in self.path
        self.render()
        return state_

    def render(self, mode='human'):
        if self.rendering:
            for i in range(self.n):
                for j in range(self.n):
                    x, y = j * self.cell_size[0], i * self.cell_size[1]
                    # current pos
                    if (i, j) == self.current_pos:
                        color = (255, 0, 0)
                    # # start pos
                    # elif (i, j) == self.start_pos:
                    #     color = (0, 255, 0)
                    # end pos
                    elif (i, j) == self.end_pos:
                        color = (0, 0, 255)
                    elif (i, j) in self.path:
                        color = (100, 155, 0)
                    # empty space
                    else:
                        color = (255, 255, 255)
                    pygame.draw.rect(self.screen, color, (x, y, self.cell_size[0], self.cell_size[1]))
            pygame.display.update()
            time.sleep(0.07)

    def step(self, action):

        self.current_step += 1
        i, j = self.current_pos

        # Update the position based on the action taken
        if action == 0:  # Down
            i = i - 1
        elif action == 2:  # Up
            i = i + 1
        elif action == 1:  # Left
            j = j - 1
        elif action == 3:  # Right
            j = j + 1

        # Check if the new position is a wall or not
        if i < 0 or i >= self.n or j < 0 or j >= self.n:
            reward = 1.
            done = True
            cause = "wall"
            cumulants = np.array([reward, 0], dtype=np.float32)
        elif (i, j) not in self.path:
            reward = 1.
            done = True
            cause = "collision"
            cumulants = np.array([reward, 0], dtype=np.float32)
        elif self.current_step >= self.max_steps:
            done = True
            cause = "slow"
            cumulants = np.array([0, 0], dtype=np.float32)
        else:
            self.current_pos = (i, j)
            new_dist_from_goal = self._calc_dist_betw_points(self.end_pos, self.current_pos)
            reward = 0. if (i, j) != self.end_pos else 1.
            if self.use_step_reward:
                reward = (self.dist_to_goal - new_dist_from_goal + reward) / 10
            self.dist_to_goal = new_dist_from_goal
            done = (i, j) == self.end_pos
            cause = None
            cumulants = np.array([0, reward], dtype=np.float32)

        state_ = self._create_obs()

        if self.video_history is not None:
            self.video_history.append(self.grid)

        info = {"cumulants": cumulants,
                "cause": cause,
                "steps": self.current_step,
                }
        if done:
            state_ = None
        reward = sum(self.default_w * cumulants)
        self.render()
        return state_, reward, done, info

    def _create_obs(self):
        self.grid = np.zeros((self.n, self.n, 3))
        for pos in self.path:
            self.grid[pos] = PATH
        # self.grid[self.start_pos] = 0.25
        self.grid[self.end_pos] = END_POS
        self.grid[self.current_pos] = CURRENT_POS

        return self.grid.flatten()

    def get_state_for_values(self, current_pos):
        self.grid = np.zeros((self.n, self.n, 3))
        for pos in self.path:
            self.grid[pos] = PATH
        # self.grid[self.start_pos] = 0.25
        self.grid[self.end_pos] = END_POS
        self.grid[current_pos] = CURRENT_POS

        return self.grid.flatten()

    def _get_current_position(self):
        return np.arange(self.n * self.n * 3).reshape((self.n, self.n, 3))[self.current_pos]

    def _calc_dist_betw_points(self, point_a, point_b):

        return abs(point_a[0] - point_b[0]) + abs(point_a[1] - point_b[1])

    def save_episode(self, path, video_name="video.avi",
                     frame_rate=10, scale_percent=10):
        if self.video_history is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            img = self.video_history[0]
            width = int(img.shape[1] * scale_percent)
            height = int(img.shape[0] * scale_percent)
            dim = (width, height)
            save_images_to_video(self.video_history, path, video_name, frame_rate, dim)

    def seed(self, seed: int = None):
        if isinstance(seed, (np.random.RandomState, Generator)):
            self.np_random = seed
            return None
        else:
            self.np_random, seed_ = seeding.np_random(seed)
            self.seed_ = self.np_random
        return seed_

    def get_max_reward(self, temp_reward):
        return np.array([1, 1])

    def set_render(self, mode, save_path=None):
        pass

    def display_text_on_gui(self, name, text=None, loc=None, rel_to_ego=True):
        pass

    def stop(self):
        pass


def main():
    n = 10
    env = GridWorld(env_config={
        "n": n,
        "default_w": [-1, 1],
        "offscreen_rendering": False,
        "video": True,
        "grid_type": "semi_static",
        "obs_type": "grid",
        "use_step_reward": True})
    pygame.init()
    for game in range(10):
        done = False
        obs = env.reset()
        env.render()
        while not done:
            events = pygame.event.get()
            action = -1
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = 1
                    elif event.key == pygame.K_RIGHT:
                        action = 3
                    elif event.key == pygame.K_UP:
                        action = 0
                    elif event.key == pygame.K_DOWN:
                        action = 2
                    else:
                        action = -1
            if action == -1:
                continue
            obs, reward, done, info = env.step(action)
            print(info, )
            if obs is not None:
                print(obs.reshape((-1, 10, 3)), )
            env.render()
        print("Reward: ", reward, info, )

    pygame.quit()


if __name__ == "__main__":
    main()
