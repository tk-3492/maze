from icecream import ic

import mujoco
import numpy as np
import torch

from framework.backends import MujocoBackend
from framework.utils import GenericTkinterViewer
from framework.sensor import PreprocessedOmniSensor, DirectionSensor
from framework.prelude import Settings, RobotLocation, Position, SensorInterface, RobotValues


class RobotNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(RobotNeuralNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(6, 3)
        self.activation1 = torch.nn.Tanhshrink()

        self.linear2 = torch.nn.Linear(3, 2)
        self.activation2 = torch.nn.Tanh()

    def forward(self, input_):
        x = self.linear1.forward(input_)
        x = self.activation1.forward(x)
        x = self.linear2.forward(x)
        x = self.activation2.forward(x)
        return x


class SampleMujocoBackend(MujocoBackend):
    def __init__(self, settings: Settings, render: bool = False):
        super().__init__(settings, render)
        self.scores = []
        self.sensors: list[tuple[SensorInterface]] = self._create_sensors()

        self.controller = RobotNeuralNetwork()
        self.input_ndarray = np.zeros((settings.Robot.NUM, 3 * 2), dtype=np.float32)
        self.input_tensor = torch.from_numpy(self.input_ndarray)

        mujoco.mj_step(self.model, self.data)

    def _create_sensors(self) -> list[tuple[SensorInterface]]:
        sensors = []
        for i, robot in enumerate(self.robot_values):
            sensor_tuple = (
                PreprocessedOmniSensor(
                    robot,
                    self.settings.Robot.ROBOT_SENSOR_GAIN,
                    self.settings.Robot.RADIUS * 2,
                    [other.site for j, other in enumerate(self.robot_values) if j != i]
                ),
                PreprocessedOmniSensor(
                    robot,
                    self.settings.Robot.FOOD_SENSOR_GAIN,
                    self.settings.Robot.RADIUS + self.settings.Food.RADIUS,
                    [food.site for food in self.food_values]
                ),
                DirectionSensor(
                    robot, self.nest_site, self.settings.Nest.RADIUS
                )
            )
            sensors.append(sensor_tuple)
        return sensors

    def _create_input_for_controller(self):
        for i, sensors in enumerate(self.sensors):
            self.input_ndarray[i, 0:2] = sensors[0].get()
            self.input_ndarray[i, 2:4] = sensors[1].get()
            self.input_ndarray[i, 4:6] = sensors[2].get()
        return self.input_tensor

    def step(self):
        with torch.no_grad():
            input_ = self._create_input_for_controller()
            output = self.controller.forward(input_)
            output_ndarray = output.numpy()

        for i, robot in enumerate(self.robot_values):
            robot.act(
                # right_wheel=output_ndarray[i, 0],
                right_wheel=1,
                # left_wheel=output_ndarray[i, 1]
                left_wheel=0
            )

        mujoco.mj_step(self.model, self.data)

    def score(self) -> list[float]:
        return self.scores

    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        if img_buf is None:
            return

        try:
            robot = self.robot_values[0]
            direction = np.array([robot.xdirection[0], robot.xdirection[1], 0])
            pos = robot.xpos
            lookat = pos + direction

            sub = pos - lookat
            self.camera.lookat[:] = lookat
            self.camera.distance = np.linalg.norm(sub)
            self.camera.azimuth = np.arctan2(
                sub[1], sub[0]
            ) * 180 / mujoco.mjPI + 180
            self.camera.elevation = -np.arcsin(
                sub[2] / self.camera.distance
            ) * 180 / mujoco.mjPI
            # self.camera.elevation = -35

            if self.do_render:
                with mujoco.Renderer(self.model, width=self.render_shape[0], height=self.render_shape[1]) as renderer:
                    renderer.update_scene(self.data, self.camera)
                    renderer.render(out=img_buf)

        except Exception as e:
            ic("MuJoCo render error:", e)
            img_buf.fill(0)


def mujoco_example():
    settings = Settings()
    settings.Render.RENDER_WIDTH = 480
    settings.Render.RENDER_HEIGHT = 320

    settings.Robot.INITIAL_POSITION = [
        RobotLocation(0, 0, np.pi / 2),
    ]
    settings.Food.INITIAL_POSITION = [
        Position(0, 2),
    ]

    viewer = GenericTkinterViewer(
        world_width=settings.Simulation.WORLD_WIDTH,
        world_height=settings.Simulation.WORLD_HEIGHT,
        render_width=settings.Render.RENDER_WIDTH,
        render_height=settings.Render.RENDER_HEIGHT,
        backend=SampleMujocoBackend(settings, render=True),
    )
    viewer.run()


if __name__ == '__main__':
    mujoco_example()
