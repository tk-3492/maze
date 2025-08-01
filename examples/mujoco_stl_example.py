import numpy as np
import torch

from framework.backends import MujocoSTL
from framework.utils import GenericTkinterViewer
from framework.prelude import Settings, RobotLocation, Position, RobotValues


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


def mujoco_stl_example():
    settings = Settings()
    settings.Render.RENDER_WIDTH = 480
    settings.Render.RENDER_HEIGHT = 320

    settings.Robot.INITIAL_POSITION = [
        RobotLocation(0, 0, np.pi / 2),
    ]
    settings.Food.INITIAL_POSITION = [
        Position(1, 2),
    ]

    viewer = GenericTkinterViewer(
        world_width=settings.Simulation.WORLD_WIDTH,
        world_height=settings.Simulation.WORLD_HEIGHT,
        render_width=settings.Render.RENDER_WIDTH,
        render_height=settings.Render.RENDER_HEIGHT,
        backend=MujocoSTL(settings, render=True),
    )
    viewer.run()


if __name__ == '__main__':
    mujoco_stl_example()
