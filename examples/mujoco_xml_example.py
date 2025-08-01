import mujoco
import numpy as np
import torch

from framework.interfaces import SimulatorBackend
from framework.utils import GenericTkinterViewer
from framework.prelude import Settings, RobotLocation, Position

WORLD_WIDTH = 10
WORLD_HEIGHT = 10
RENDER_WIDTH = 640
RENDER_HEIGHT = 480

xml = f"""
<mujoco>
  <asset>
    <texture name="simple_checker" type="2d" builtin="checker" width="256" height="256" rgb1="1.0 1.0 1.0" rgb2="0.7 0.7 0.7"/>
    <material name="ground" texture="simple_checker" texrepeat="{WORLD_WIDTH * 0.5} {WORLD_HEIGHT * 0.5}" />
  </asset>

  <worldbody>
    <geom type="plane" size="{WORLD_WIDTH} {WORLD_HEIGHT} 1" material="ground"/>
    
    <body name="car" pos="0 0 3" axisangle="1 1 1 30">
      <freejoint/>
      <geom name="sample_box" type="box" size="1 1.2 1.5" rgba="1 0.5 0 0.5" />
    </body>
  </worldbody>

</mujoco>
"""


class XmlExampleBackend(SimulatorBackend):
    def __init__(self, render: bool = False):
        self._do_render = render
        self.render_shape = (RENDER_WIDTH, RENDER_HEIGHT)
        self.camera = mujoco.MjvCamera()

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        mujoco.mj_step(self.model, self.data)

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

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def get_scores(self) -> list[float]:
        return []

    def calc_total_score(self) -> float:
        return 0


def mujoco_example():
    viewer = GenericTkinterViewer(
        world_width=WORLD_WIDTH,
        world_height=WORLD_HEIGHT,
        render_width=RENDER_WIDTH,
        render_height=RENDER_HEIGHT,
        backend=XmlExampleBackend(render=True),
    )
    viewer.run()


if __name__ == '__main__':
    mujoco_example()
