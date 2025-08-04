from framework.utils import GenericTkinterViewer

from utils import Simulator
from settings import MySettings


def main():
    settings = MySettings()

    viewer = GenericTkinterViewer(
        world_width=settings.Simulation.WORLD_WIDTH,
        world_height=settings.Simulation.WORLD_HEIGHT,
        render_width=settings.Render.RENDER_WIDTH,
        render_height=settings.Render.RENDER_HEIGHT,
        backend=Simulator(settings, render=True),
    )
    viewer.run()


if __name__ == '__main__':
    main()
