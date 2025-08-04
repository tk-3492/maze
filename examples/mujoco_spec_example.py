#
# このコードはMjSpecを使用してMuJoCoのシミュレーション環境を生成する例です．
# 同様の環境を生成するXMLはmujoco_xml_example.pyにあり，実行結果はmujoco_xml_example.pyと一致します．
#

import mujoco
import numpy as np

from framework.interfaces import SimulatorBackend
from framework.utils import GenericTkinterViewer
from framework.environment import add_texture, add_material, add_geom, add_body, axisangle_to_quat, add_joint
from framework.prelude import Settings

WORLD_WIDTH = 10
WORLD_HEIGHT = 10
RENDER_WIDTH = 640
RENDER_HEIGHT = 480


def generate_spec():
    spec = mujoco.MjSpec()  # <mujoco> ~ </mujoco>

    # <texture name="simple_checker" type="2d" builtin="checker" width="256" height="256" rgb1="1.0 1.0 1.0" rgb2="0.7 0.7 0.7"/>
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

    # <material name="ground" texture="simple_checker" texrepeat="{WORLD_WIDTH * 0.5} {WORLD_HEIGHT * 0.5}" />
    add_material(
        spec,
        name="ground",
        texture="simple_checker",
        texrepeat=(
            WORLD_WIDTH * 0.5,
            WORLD_HEIGHT * 0.5
        )
    )

    # <worldbody> ~ </worldbody>
    worldbody = spec.worldbody

    # <worldbody>
    #   <geom type="plane" size="{WORLD_WIDTH} {WORLD_HEIGHT} 1" material="ground"/>
    # </worldbody>
    add_geom(
        worldbody,
        geom_type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0, 0, 0),
        size=(WORLD_WIDTH, WORLD_HEIGHT, 1),
        material="ground",
    )

    # <worldbody>
    #   <body name="car" pos="0 0 3" axisangle="1 1 1 30"> ~ </body>
    # </worldbody>
    cube_body = add_body(
        spec.worldbody,
        pos=(0, 0, 3),
        quat=axisangle_to_quat(axis=(1, 1, 1), angle=mujoco.mjPI / 6)
    )

    # <body>
    #   <freejoint/>
    # </body>
    add_joint(
        cube_body,
        name="cube_freejoint",
        joint_type=mujoco.mjtJoint.mjJNT_FREE,
    )

    # <body>
    #   <geom name="sample_box" type="box" size="1 1.2 1.5" rgba="1 0.5 0 0.5" />
    # </body>
    add_geom(
        cube_body,
        name="sample_box",
        geom_type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(1, 1.2, 1.5),
        rgba=(1, 0.5, 0, 0.5),
    )

    return spec


class SpecExampleBackend(SimulatorBackend):
    def __init__(self, render: bool = False):
        self._do_render = render
        self.render_shape = (RENDER_WIDTH, RENDER_HEIGHT)
        self.camera = mujoco.MjvCamera()

        self.spec = generate_spec()
        self.model = self.spec.compile()
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
        backend=SpecExampleBackend(render=True),
    )
    viewer.run()


if __name__ == '__main__':
    mujoco_example()
