def viewer_example():
    from framework.backends import SimpleAnimatedBackend
    from framework.utils import GenericTkinterViewer
    from framework.prelude import Settings

    settings = Settings()
    settings.Render.RENDER_WIDTH = 480
    settings.Render.RENDER_HEIGHT = 320

    viewer = GenericTkinterViewer(
        world_width=settings.Simulation.WORLD_WIDTH,
        world_height=settings.Simulation.WORLD_HEIGHT,
        render_width=settings.Render.RENDER_WIDTH,
        render_height=settings.Render.RENDER_HEIGHT,
        backend=SimpleAnimatedBackend(settings),
    )
    viewer.run()


if __name__ == '__main__':
    viewer_example()
