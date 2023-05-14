""" This module contains the Navigation Track environment class for Ray Casting """
import copy
import math
from typing import Optional, Tuple

import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete
from pygame import Surface

from gym_navigation.enums.action import Action
from gym_navigation.enums.color import Color
from gym_navigation.envs.navigation import Navigation
from gym_navigation.geometry.line import Line, NoIntersectionError
from gym_navigation.geometry.point import Point
from gym_navigation.geometry.pose import Pose


class NavigationTrack(Navigation):
    SHIFT_STANDARD_DEVIATION = 0.02
    SENSOR_STANDARD_DEVIATION = 0.02
    SHIFT = 0.2

    COLLISION_THRESHOLD = 0.4  # Robot radius
    COLLISION_REWARD = -100.0
    FORWARD_REWARD = 2

    SCAN_ANGLES = (
        -math.pi / 8,
        -math.pi / 4,
        -3 * math.pi / 8,
        -math.pi / 2,
        -5 * math.pi / 8,
        -3 * math.pi / 4,
        -7 * math.pi / 8,
        0,
        math.pi / 8,
        math.pi / 4,
        3 * math.pi / 8,
        math.pi / 2,
        5 * math.pi / 8,
        3 * math.pi / 4,
        7 * math.pi / 8,
        math.pi
    )  # angles of ray casting

    SCAN_RANGE_MAX = 15.0  # 17.0
    SCAN_RANGE_MIN = 0.0
    NUMBER_OF_RAYS = len(SCAN_ANGLES)  # 16
    NUMBER_OF_OBSERVATIONS = NUMBER_OF_RAYS

    pose: Pose
    ranges: np.ndarray

    def __init__(self, render_mode=None, track_id: int = 1) -> None:
        super().__init__(render_mode, track_id)

        self.ranges = np.empty(self.NUMBER_OF_RAYS)

        self.action_space = Discrete(len(Action))

        self.observation_space = Box(low=self.SCAN_RANGE_MIN,
                                     high=self.SCAN_RANGE_MAX,
                                     shape=(self.NUMBER_OF_OBSERVATIONS,),
                                     dtype=np.float64)

    def perform_action(self, action: int) -> None:
        action_enum = Action(action)
        theta = (self.np_random.normal(0, self.SHIFT_STANDARD_DEVIATION)
                 + action_enum.angular_shift)
        distance = (self.np_random.normal(0, self.SHIFT_STANDARD_DEVIATION)
                    + action_enum.linear_shift)
        self.pose.rotate(theta)
        self.pose.move(distance)
        # self.pose.shift(distance, theta)
        self.update_scan()

    def update_scan(self) -> None:
        scan_lines = self.create_scan_lines()
        for i, scan_line in enumerate(scan_lines):
            min_distance = self.SCAN_RANGE_MAX
            for wall in self.world:
                try:
                    intersection = scan_line.get_intersection(wall)
                except NoIntersectionError:
                    continue

                distance = self.pose.position.calculate_distance(intersection)
                if distance < min_distance:
                    min_distance = distance

            # Gaussian distribution
            sensor_noise = self.np_random.normal(0, self.SENSOR_STANDARD_DEVIATION)
            self.ranges[i] = min_distance + sensor_noise
            # distance by i-th ray = min distance plus sensor precision error

    # Creates 16 Lines which start in the robot position and end in scan_pose moved by SCAN_RANGE_MAX
    def create_scan_lines(self) -> np.ndarray:
        scan_poses = self.create_scan_poses()
        scan_lines = np.empty(self.NUMBER_OF_RAYS, dtype=Line)

        for i, scan_pose in enumerate(scan_poses):
            scan_pose.move(self.SCAN_RANGE_MAX)
            scan_lines[i] = Line(copy.copy(self.pose.position),
                                 scan_pose.position)

        return scan_lines

    # Creates 16 Poses with robot position and scan angle yaw
    def create_scan_poses(self) -> np.ndarray:
        scan_poses = np.empty(self.NUMBER_OF_RAYS, dtype=Pose)

        for i, scan_angle in enumerate(self.SCAN_ANGLES):
            scan_poses[i] = Pose(copy.copy(self.pose.position),
                                 self.pose.yaw + scan_angle)

        return scan_poses

    def get_observation(self) -> np.ndarray:
        return self.ranges.copy()  # return ray measured distances

    def check_if_truncated(self) -> bool:
        return False

    def check_if_terminated(self) -> bool:
        return self.collision_occurred()

    def collision_occurred(self) -> bool:
        return bool((self.ranges < self.COLLISION_THRESHOLD).any())

    def calculate_reward(self) -> float:
        if self.collision_occurred():
            reward = self.COLLISION_REWARD
        else:
            reward = self.FORWARD_REWARD

        return reward

    def init_environment(self, options: Optional[dict] = None) -> None:
        self.init_pose()
        self.update_scan()

    def create_info(self, goal_reached: bool = False) -> dict:
        return {}

    def init_pose(self) -> None:
        area = self.np_random.choice(self.track.spawn_area)
        x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
        y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
        position = Point(x_coordinate, y_coordinate)
        yaw = self.np_random.uniform(-math.pi, math.pi)  # Random initial orientation
        self.pose = Pose(position, yaw)

    def draw(self, canvas: Surface) -> None:
        # CANVAS
        canvas.fill(Color.WHITE.value)

        # WALL AND OBSTACLES
        for wall in self.world:
            pygame.draw.line(canvas,
                             Color.BLACK.value,
                             self.convert_point(wall.start),
                             self.convert_point(wall.end),
                             self.WIDTH)

        # RAYS
        # uso min distance per disegnare rays
        scan_poses = self.create_scan_poses()
        for i, scan_pose in enumerate(scan_poses):
            scan_pose.move(self.ranges[i])
            pygame.draw.line(canvas,
                             Color.YELLOW.value,
                             self.convert_point(self.pose.position),
                             self.convert_point(scan_pose.position),
                             self.WIDTH)

        # ROBOT
        pygame.draw.circle(canvas,
                           Color.BLUE.value,
                           self.convert_point(self.pose.position),
                           self.COLLISION_THRESHOLD * self.RESOLUTION)

    def convert_point(self, point: Point) -> Tuple[int, int]:
        pygame_x = (round(point.x_coordinate * self.RESOLUTION)
                    + self.X_OFFSET)
        pygame_y = (self.window_size
                    - round(point.y_coordinate * self.RESOLUTION)
                    + self.Y_OFFSET)
        return pygame_x, pygame_y
