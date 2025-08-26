import numpy as np

from mujoco._structs import _MjDataSiteViews

from ..prelude import *


class DirectionSensor(SensorInterface):
    def __init__(self, robot: RobotValues, target_site: _MjDataSiteViews, target_radius: float):
        self.robot_values = robot
        self.target = target_site
        self.target_radius = target_radius

    def get(self) -> np.ndarray:
        direction = self.robot_values.xdirection[:2]
        center = self.robot_values.xpos[:2]
        target = self.target.xpos[0:2]

        direction_matrix = np.array([
            [direction[1], -direction[0]],
            [direction[0], direction[1]],
        ])

        sub = target - center
        distance = np.linalg.norm(sub)

        if distance <= 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)

        normalized_sub = sub / distance

        # k = 1 / max(distance, self.target_radius)
        # result = np.dot(direction_matrix, sub * k)

        result = np.dot(direction_matrix, normalized_sub)

        angle = np.arctan2(-result[0], result[1]) / np.pi
        magnitude = distance / self.target_radius if distance < self.target_radius else 1.0

        return np.array([magnitude, angle], dtype=np.float32)
