""" This module contains the general navigation class. """

from abc import abstractmethod
from copy import deepcopy
from typing import Tuple

import numpy as np
import pygame
from gymnasium import Env
from pygame import Surface

from gym_navigation.enums.track import Track
from gym_navigation.geometry.line import Line


class Navigation(Env):
    """
    This class is used to define the step, reset, render, close methods
    as template methods. In this way we can create multiple environments that
    can inherit from one another and only redefine certain methods.
    """

    RESOLUTION = 20  # 1.0 => 20 pixels
    WIDTH = 3
    X_OFFSET = 250  # 150
    Y_OFFSET = -250  # -150

    track: Track
    world: Tuple[Line, ...]
    reward: int

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, track_id: int = 1) -> None:
        self.window_size = 700  # The size of the PyGame window
        self.track = Track(track_id)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.steps = 0

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:

        assert self.action_space.contains(action)
        self.steps += 1
        self.perform_action(action)

        observation = self.get_observation()
        terminated = self.check_if_terminated()

        if self.steps >= 100:
            self.reward = -100
            truncated = True
        else:
            self.reward = self.calculate_reward()
            truncated = False

        if self.render_mode == "human":
            self.render_frame()

        return observation, self.reward, terminated, truncated, {}

    @abstractmethod
    def perform_action(self, action) -> None:
        pass

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        pass

    @abstractmethod
    def check_if_terminated(self) -> bool:
        pass

    @abstractmethod
    def check_if_truncated(self) -> bool:
        """
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        """
        pass

    @abstractmethod
    def calculate_reward(self) -> float:
        pass

    @abstractmethod
    def create_info(self, goal_reached: bool = False) -> dict:
        pass

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:

        super().reset(seed=seed)
        self.world = deepcopy(self.track.walls)

        self.init_environment(options)

        observation = self.get_observation()
        info = self.create_info()
        self.reward = 0
        self.steps = 0

        if self.render_mode == "human":
            self.render_frame()

        return observation, info

    @abstractmethod
    def init_environment(self, options=None) -> None:
        pass

    def render(self):
        return None

    def render_frame(self) -> None:
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.window is None or self.clock is None:
            return

        canvas = Surface((self.window_size, self.window_size))
        # Draw the canvas
        self.draw(canvas)

        if self.render_mode == "human":
            # The following line copies our drawings from canvas to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            self.clock.tick(self.metadata["render_fps"])

    @abstractmethod
    def draw(self, canvas: Surface) -> None:
        pass

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
