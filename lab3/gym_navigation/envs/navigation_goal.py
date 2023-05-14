""" This module contains the Navigation Goal environment class """
import math
from typing import Optional

import numpy as np
import pygame
from gymnasium.spaces import Box
from pygame import Surface

from gym_navigation.enums.color import Color
from gym_navigation.envs.navigation_track import NavigationTrack
from gym_navigation.geometry.line import Line
from gym_navigation.geometry.point import Point


class NavigationGoal(NavigationTrack):
    GOAL_THRESHOLD = 1
    MINIMUM_DISTANCE_OBSTACLE_GOAL = 2.5
    MINIMUM_DISTANCE_OBSTACLE_ROBOT = 1.5
    MINIMUM_DISTANCE_ROBOT_GOAL = 4
    MAXIMUM_DISTANCE_OBSTACLE_GOAL = 6
    MAXIMUM_DISTANCE_ROBOT_GOAL = 8

    DISTANCE_STANDARD_DEVIATION = 0.000001
    ANGLE_STANDARD_DEVIATION = 0.000001

    GOAL_REWARD = 500.0  # if you change this change also in navigation.py step() method
    BACKWARD_REWARD = -1.0

    NUMBER_OF_OBSERVATIONS = NavigationTrack.NUMBER_OF_RAYS + 2
    N_OBSTACLES = 3  # 0 1 2 3 4
    OBSTACLES_LENGTH = 1

    previous_distance_from_goal: float
    distance_from_goal: float
    angle_from_goal: float
    goal: Point

    def __init__(self,
                 render_mode=None, track_id: int = 1) -> None:
        super().__init__(render_mode, track_id)

        high = np.array(self.NUMBER_OF_RAYS * [self.SCAN_RANGE_MAX] + [self.SCAN_RANGE_MAX] + [math.pi],
                        dtype=np.float64)
        low = np.array(self.NUMBER_OF_RAYS * [self.SCAN_RANGE_MIN] + [0.0] + [0.0], dtype=np.float64)

        self.observation_space = Box(low=low,
                                     high=high,
                                     shape=(self.NUMBER_OF_OBSERVATIONS,),
                                     dtype=np.float64)

    def perform_action(self, action: int) -> None:
        super().perform_action(action)
        distance_noise = self.np_random.normal(0, self.DISTANCE_STANDARD_DEVIATION)
        self.distance_from_goal = distance_noise + self.pose.position.calculate_distance(self.goal)
        angle_noise = self.np_random.normal(0, self.ANGLE_STANDARD_DEVIATION)
        self.angle_from_goal = angle_noise + self.pose.calculate_angle_difference(self.goal)
        if self.angle_from_goal < 0:
            self.angle_from_goal = math.pi + self.angle_from_goal

    # return ray cast distances and direction and angle of the goal
    def get_observation(self) -> np.ndarray:
        return np.append(self.ranges.copy(), [self.distance_from_goal, self.angle_from_goal])

    def check_if_terminated(self) -> bool:
        return self.collision_occurred() or self.goal_reached()

    def goal_reached(self) -> bool:
        return self.distance_from_goal < self.GOAL_THRESHOLD

    def calculate_reward(self) -> float:
        if self.collision_occurred():
            reward = self.COLLISION_REWARD
        elif self.goal_reached():
            reward = self.GOAL_REWARD
        elif self.distance_from_goal < self.previous_distance_from_goal:
            if 0.0 <= self.angle_from_goal <= math.pi / 4:
                reward = self.FORWARD_REWARD * (self.angle_from_goal / (math.pi / 4))
                reward = 2.0 - reward
            elif 3 * math.pi / 4 <= self.angle_from_goal <= math.pi:
                angle = math.pi - self.angle_from_goal
                reward = self.FORWARD_REWARD * (angle / (math.pi / 4))
                reward = 2.0 - reward
            else:
                reward = 0
        else:
            if math.pi / 4 <= self.angle_from_goal <= math.pi / 2:
                angle = math.pi / 2 - self.angle_from_goal
                reward = angle / (math.pi / 4)
                reward = (1.0 - reward) * self.BACKWARD_REWARD
            elif math.pi / 2 <= self.angle_from_goal <= 3 * math.pi / 4:
                angle = 3 * math.pi / 4 - self.angle_from_goal
                reward = self.BACKWARD_REWARD * (angle / (math.pi / 4))
            else:
                reward = 0.0

        self.previous_distance_from_goal = self.distance_from_goal
        return reward

    def init_environment(self, options: Optional[dict] = None) -> None:
        self.init_pose()
        self.init_goal()
        self.init_obstacles()
        self.update_scan()

    def init_goal(self) -> None:
        while True:
            area = self.np_random.choice(self.track.spawn_area)
            x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
            y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
            goal = Point(x_coordinate, y_coordinate)
            distance_from_pose = goal.calculate_distance(self.pose.position)
            if self.MINIMUM_DISTANCE_ROBOT_GOAL <= distance_from_pose <= self.MAXIMUM_DISTANCE_ROBOT_GOAL:
                break

        self.goal = goal
        distance_noise = self.np_random.normal(0, self.DISTANCE_STANDARD_DEVIATION)
        self.distance_from_goal = distance_noise + self.pose.position.calculate_distance(self.goal)
        self.previous_distance_from_goal = self.distance_from_goal
        angle_noise = self.np_random.normal(0, self.ANGLE_STANDARD_DEVIATION)
        self.angle_from_goal = angle_noise + self.pose.calculate_angle_difference(self.goal)
        if self.angle_from_goal < 0:
            self.angle_from_goal = math.pi + self.angle_from_goal

    def init_obstacles(self) -> None:
        for _ in range(self.N_OBSTACLES):
            while True:
                area = self.np_random.choice(self.track.spawn_area)
                x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
                y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
                obstacles_center = Point(x_coordinate, y_coordinate)
                distance_from_pose = obstacles_center.calculate_distance(self.pose.position)
                distance_from_goal = obstacles_center.calculate_distance(self.goal)

                # If Agent can spawn near obstacles
                # if (distance_from_pose > self.MINIMUM_DISTANCE_OBSTACLE_ROBOT or distance_from_goal < self.MAXIMUM_DISTANCE_OBSTACLE_GOAL) and distance_from_goal > self.MINIMUM_DISTANCE_OBSTACLE_GOAL:
                if distance_from_pose > self.MINIMUM_DISTANCE_OBSTACLE_ROBOT and self.MAXIMUM_DISTANCE_OBSTACLE_GOAL > distance_from_goal > self.MINIMUM_DISTANCE_OBSTACLE_GOAL:
                    break

            point1 = Point(
                obstacles_center.x_coordinate - self.OBSTACLES_LENGTH / 2,
                obstacles_center.y_coordinate - self.OBSTACLES_LENGTH / 2)
            point2 = Point(
                obstacles_center.x_coordinate - self.OBSTACLES_LENGTH / 2,
                obstacles_center.y_coordinate + self.OBSTACLES_LENGTH / 2)
            point3 = Point(
                obstacles_center.x_coordinate + self.OBSTACLES_LENGTH / 2,
                obstacles_center.y_coordinate + self.OBSTACLES_LENGTH / 2)
            point4 = Point(
                obstacles_center.x_coordinate + self.OBSTACLES_LENGTH / 2,
                obstacles_center.y_coordinate - self.OBSTACLES_LENGTH / 2)

            self.world += (Line(point1, point2),)
            self.world += (Line(point2, point3),)
            self.world += (Line(point3, point4),)
            self.world += (Line(point4, point1),)

    def draw(self, canvas: Surface) -> None:
        super().draw(canvas)
        # GOAL
        pygame.draw.circle(canvas,
                           Color.GREEN.value,
                           self.convert_point(self.goal),
                           self.GOAL_THRESHOLD * self.RESOLUTION)

    def create_info(self, goal_reached: bool = False) -> dict:
        if goal_reached:
            return {"result": "Goal_Reached"}
        else:
            return {"result": "Failed"}
