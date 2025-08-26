import numpy as np

from mujoco._structs import _MjDataSiteViews

from icecream import ic
from ..prelude import *


class OmniSensor(SensorInterface):
    """
    三角関数を用いた全方位センサです．

    登録された知覚対象のMjDataSiteViewsから対象の位置を取得し，センサーの出力を計算します．
    """

    def __init__(
            self,
            robot: RobotValues,
            o_gain: float,
            d_gain: float,
            offset: float,
            target_sites: list[_MjDataSiteViews]
    ):
        """
        OmniSensorのコンストラクタ。

        Args:
            o_gain (float): 最終的な出力にかけられる値．
            d_gain (float): 近く対象との距離にかかわるセンサーのゲイン値。
            offset (float): センサーのオフセット値。
            target_sites (list[_MjDataSiteViews]): ターゲットサイトのリスト。
        """
        self.robot = robot
        self.o_gain = o_gain
        self.d_gain = d_gain
        self.offset = offset
        self.targets: list[_MjDataSiteViews] = target_sites
        self.target_positions = np.zeros((len(self.targets), 2))

    def _update_positions(self):
        for i, p in enumerate(self.targets):
            self.target_positions[i] = p.xpos[0:2]

    def get(self) -> np.ndarray:
        if len(self.targets) == 0:
            return np.zeros(2)

        bot_direction = self.robot.xdirection[:2]
        bot_pos = self.robot.xpos[:2]

        direction = np.array([
            [bot_direction[1], -bot_direction[0]],
            [bot_direction[0], bot_direction[1]],
        ])
        self._update_positions()

        sub = self.target_positions - bot_pos
        distance = np.linalg.norm(sub, axis=1)
        if np.any(distance == 0):
            # ic("distance array contains 0")
            return np.zeros(2)

        sub = sub.T / distance
        trigono_components = np.dot(direction, sub)
        scaled_distance = self.d_gain * np.maximum(distance - self.offset, 0)
        res = np.dot(
            trigono_components,
            np.reciprocal(scaled_distance + 1),
        )

        return res * self.o_gain
