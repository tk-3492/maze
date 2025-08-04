import mujoco
import numpy as np

from framework.interfaces import SimulatorBackend
from framework.environment import add_texture, add_material, add_geom
from framework.prelude import Settings


def generate_spec(settings: Settings):
    spec = mujoco.MjSpec()

    visual: mujoco._specs.MjVisual = spec.visual
    visual.global_.offwidth = settings.Render.RENDER_WIDTH
    visual.global_.offheight = settings.Render.RENDER_HEIGHT

    add_texture(
        spec,
        name="simple_checker",
        type_=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=256,
        height=256,
        rgb1=(1.0, 1.0, 1.0),
        rgb2=(0.7, 0.7, 0.7)
    )
    add_material(
        spec,
        name="ground",
        texture="simple_checker",
        texrepeat=(
            settings.Simulation.WORLD_WIDTH * 0.5,
            settings.Simulation.WORLD_HEIGHT * 0.5
        )
    )

    add_geom(
        spec.worldbody,
        geom_type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0, 0, 0),
        size=(settings.Simulation.WORLD_WIDTH * 0.5, settings.Simulation.WORLD_HEIGHT * 0.5, 1),
        material="ground",
    )

    rng = np.random.default_rng()
    xs = rng.uniform(
        -settings.Simulation.WORLD_WIDTH * 0.5, settings.Simulation.WORLD_WIDTH * 0.5, (3,)
    )
    ys = rng.uniform(
        -settings.Simulation.WORLD_HEIGHT * 0.5, settings.Simulation.WORLD_HEIGHT * 0.5, (3,)
    )
    for x, y in zip(xs, ys):
        add_geom(
            spec.worldbody,
            geom_type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.5, 0.5, 0.5),  # BoxであってもSizeは中心からの半径のように設定するため1m x 1m x 1mは(0.5, 0.5, 0.5)になる
            rgba=(0.7, 0.7, 0.7, 1.0),
            pos=(x, y, 0.5),
        )

    return spec


class Simulator(SimulatorBackend):
    def __init__(self, settings: Settings, render: bool = False):
        self.settings = settings
        self._do_render = render

        self.render_shape = (settings.Render.RENDER_WIDTH, settings.Render.RENDER_HEIGHT)
        self.camera = mujoco.MjvCamera()

        self.spec = generate_spec(settings)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        mujoco.mj_step(self.model, self.data)

    def reset(self):
        self.spec = generate_spec(self.settings)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        if img_buf is None:
            return

        try:
            pos = np.array(pos)
            lookat = np.array(lookat)
            sub = pos - lookat
            self.camera.lookat[:] = lookat
            self.camera.distance = np.linalg.norm(sub)
            self.camera.azimuth = np.arctan2(
                sub[1], sub[0]
            ) * 180 / mujoco.mjPI + 180
            self.camera.elevation = -np.arcsin(
                sub[2] / self.camera.distance
            ) * 180 / mujoco.mjPI

            if self._do_render:
                with mujoco.Renderer(self.model, width=self.render_shape[0], height=self.render_shape[1]) as renderer:
                    renderer.update_scene(self.data, self.camera)
                    renderer.render(out=img_buf)

        except Exception as e:
            img_buf.fill(0)

    def get_scores(self) -> list[float]:
        return []

    def calc_total_score(self) -> float:
        return 0
