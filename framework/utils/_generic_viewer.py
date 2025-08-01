import dataclasses
import enum
import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import Optional
import time
import threading
import logging
from PIL import Image, ImageTk

from ..prelude import *


class SimulationRunningMode(enum.Enum):
    RUNNING = "running"
    PAUSED = "paused"
    RESTART = "restart"
    ONE_STEP = "one_step"


@dataclasses.dataclass
class WidthAndHeight:
    WORLD_WIDTH: float
    WORLD_HEIGHT: float
    RENDER_WIDTH: int
    RENDER_HEIGHT: int


class _SimulationState:
    def __init__(self, wh: WidthAndHeight):
        self.rgb_buffer: np.ndarray = np.zeros((wh.RENDER_HEIGHT, wh.RENDER_WIDTH, 3), dtype=np.uint8)
        self.buffer_update_timestamp: Optional[float] = time.time()

        self.running_mode: SimulationRunningMode = SimulationRunningMode.RUNNING
        self.mode_change_event: threading.Event = threading.Event()

        self.lookat_position: Position3d = Position3d(DEFAULT_LOOKAT_X, DEFAULT_LOOKAT_Y, DEFAULT_LOOKAT_Z)
        self.camera_position: Position3d = Position3d(0, 0, 0)
        self.set_camera_position_from_angles(DEFAULT_CAMERA_DISTANCE, DEFAULT_CAMERA_AZIMUTH, DEFAULT_CAMERA_ELEVATION)
        self.fps: float = 0.0

        self.error_message: Optional[str] = None
        self.error_timestamp: Optional[float] = None

    def update_running_mode_by_external(self, order: SimulationRunningMode):
        if self.running_mode != order:
            self.running_mode = order  # 完全な同期処理を目指していないため読み取りにおける競合は無視してよい

            for _ in range(10):
                if self.mode_change_event.wait(timeout=0.1):
                    break
            else:
                logging.getLogger(__name__).warning(
                    f"Timeout waiting for mode change to {order}, resetting event"
                )
            self.mode_change_event.clear()

        return self.running_mode

    def set_error(self, message: str):
        self.error_message = message
        self.error_timestamp = time.time()

    def clear_error(self):
        self.error_message = None
        self.error_timestamp = None

    def set_camera_position_from_angles(self, distance: float, azimuth_deg: float, elevation_deg: float):
        azimuth_rad = np.radians(azimuth_deg)
        elevation_rad = np.radians(elevation_deg)

        x_offset = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y_offset = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z_offset = distance * np.sin(elevation_rad)

        self.camera_position = Position3d(
            self.lookat_position.x + x_offset,
            self.lookat_position.y + y_offset,
            self.lookat_position.z + z_offset
        )

    @property
    def has_recent_error(self) -> bool:
        return (self.error_message is not None and
                self.error_timestamp is not None and
                time.time() - self.error_timestamp < 5.0)


class _Simulation:
    def __init__(self, backend: SimulatorBackend, state: _SimulationState):
        self.backend = backend
        self.state = state
        self.logger = logging.getLogger(__name__)

        self.thread: Optional[threading.Thread] = None
        self.should_stop = False
        self._running = False
        self._average_elapsed = 0.0

    def start(self):
        if not self._running:
            self.should_stop = False
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            self._running = True

    def stop(self):
        if self._running:
            self.should_stop = True
            if self.thread:
                self.logger.info("Stopping simulation thread")
                self.thread.join(timeout=THREAD_JOIN_TIMEOUT)
                if self.thread.is_alive():
                    self.logger.warning("Simulation thread did not stop gracefully")
            self._running = False

    def _render_frame(self):
        try:
            self.backend.render(
                self.state.rgb_buffer,
                self.state.camera_position.to_tuple(),
                self.state.lookat_position.to_tuple()
            )
            self.state.buffer_update_timestamp = time.time()
        except Exception as e:
            self.logger.error(f"Render error: {e}")
            self.state.set_error(f"Render error: {e}")
            self.state.update_running_mode_by_external(SimulationRunningMode.PAUSED)

    def _run_loop(self):
        target_interval = 1.0 / TARGET_FPS
        running_mode = self.state.running_mode

        try:
            while not self.should_stop:
                start_time = time.perf_counter()

                if running_mode != self.state.running_mode:
                    match self.state.running_mode:
                        case SimulationRunningMode.RESTART:
                            self.backend.reset()
                            running_mode = SimulationRunningMode.PAUSED

                        case SimulationRunningMode.ONE_STEP:
                            self.backend.step()
                            running_mode = SimulationRunningMode.PAUSED

                        case SimulationRunningMode.RUNNING:
                            running_mode = SimulationRunningMode.RUNNING

                        case SimulationRunningMode.PAUSED:
                            running_mode = SimulationRunningMode.PAUSED

                    self.state.running_mode = running_mode
                    self.state.mode_change_event.set()

                elif running_mode == SimulationRunningMode.RUNNING:
                    self.backend.step()

                sleep_time = max(0.0, target_interval - self._average_elapsed)
                if sleep_time > 0:
                    self._render_frame()
                    time.sleep(sleep_time)

                elapsed = time.perf_counter() - start_time
                self._average_elapsed = (self._average_elapsed * 0.9) + (elapsed * 0.1)

                if self._average_elapsed > 0:
                    self.state.fps = 1.0 / self._average_elapsed


        except Exception as e:
            self.logger.critical(f"Critical error in simulation loop: {e}")
            self.state.set_error(f"Critical simulation error: {e}")


class SimulationControlPanel(tk.Frame):
    def __init__(self, parent, state: _SimulationState):
        super().__init__(parent)
        self.state = state

        # Set up UI elements
        ttk.Label(self, text="Simulation Control", font=('Arial', 12, 'bold')).pack(pady=(0, 10))

        self.pause_button = ttk.Button(self, text="Pause")
        self.pause_button.pack(fill=tk.X, pady=2)

        step_button = ttk.Button(self, text="Step")
        step_button.pack(fill=tk.X, pady=2)

        reset_button = ttk.Button(self, text="Reset")
        reset_button.pack(fill=tk.X, pady=2)

        # Set up callbacks
        self.pause_button.config(command=self._handle_pause)
        step_button.config(command=self._handle_step)
        reset_button.config(command=self._handle_reset)

        self._update_pause_button_text()

    def _update_pause_button_text(self):
        if self.state.running_mode == SimulationRunningMode.RUNNING:
            self.pause_button.config(text="Pause")
        else:
            self.pause_button.config(text="Resume")

    def _handle_pause(self):
        if self.state.running_mode == SimulationRunningMode.RUNNING:
            self.state.update_running_mode_by_external(SimulationRunningMode.PAUSED)
        else:
            self.state.update_running_mode_by_external(SimulationRunningMode.RUNNING)
        self._update_pause_button_text()

    def _handle_step(self):
        self.state.update_running_mode_by_external(SimulationRunningMode.ONE_STEP)

    def _handle_reset(self):
        self.state.update_running_mode_by_external(SimulationRunningMode.RESTART)
        self.state.clear_error()

    def update(self):
        self._update_pause_button_text()


class CameraControlPanel(ttk.Frame):
    def __init__(self, parent_frame: ttk.Frame, wh: WidthAndHeight, state: _SimulationState):
        super().__init__(parent_frame)
        self.state = state

        self.lookat_x_var = tk.DoubleVar(value=DEFAULT_LOOKAT_X)
        self.lookat_y_var = tk.DoubleVar(value=DEFAULT_LOOKAT_Y)

        # Spherical coordinates for camera control
        self.distance_var = tk.DoubleVar(value=DEFAULT_CAMERA_DISTANCE)
        self.azimuth_var = tk.DoubleVar(value=DEFAULT_CAMERA_AZIMUTH)
        self.elevation_var = tk.DoubleVar(value=DEFAULT_CAMERA_ELEVATION)

        # Panel title
        ttk.Label(self, text="Camera Control", font=('Arial', 12, 'bold')).pack(pady=(0, 10))

        # LookAt controls
        ttk.Label(self, text="Look At Position:", font=('Arial', 10, 'bold')).pack(pady=(10, 5))

        def _install_scale_helper(var_, range_min_, range_max_, label_):
            ttk.Label(self, text=label_).pack()
            pos_scale = ttk.Scale(
                self, from_=range_min_, to=range_max_, variable=var_, orient=tk.HORIZONTAL,
                command=self._handle_camera_change
            )
            pos_scale.pack(fill=tk.X, pady=2)

        for label, var, range_min, range_max in [
            ("X:", self.lookat_x_var, -wh.WORLD_WIDTH * 0.5, wh.WORLD_WIDTH * 0.5),
            ("Y:", self.lookat_y_var, -wh.WORLD_HEIGHT * 0.5, wh.WORLD_HEIGHT * 0.5),
        ]:
            _install_scale_helper(var, range_min, range_max, label)

        # Spherical camera controls
        ttk.Label(self, text="Camera Position (Spherical):", font=('Arial', 10, 'bold')).pack(pady=(15, 5))

        for label, var, range_min, range_max in [
            ("Distance:", self.distance_var, 1.0, 20.0),
            ("Azimuth (°):", self.azimuth_var, -270.0, 90.0),
            ("Elevation (°):", self.elevation_var, 0.0, 90.0)
        ]:
            _install_scale_helper(var, range_min, range_max, label)

    def _handle_camera_change(self, value=None):
        self.state.lookat_position = Position3d(
            self.lookat_x_var.get(),
            self.lookat_y_var.get(),
            DEFAULT_LOOKAT_Z
        )

        self.state.set_camera_position_from_angles(
            self.distance_var.get(),
            self.azimuth_var.get(),
            self.elevation_var.get()
        )


class SimulationInfoPanel(ttk.LabelFrame):
    def __init__(self, parent, state: _SimulationState, backend_name: str = "Unknown"):
        super().__init__(parent, text="Simulation Info")
        self.state = state

        self.fps_label = ttk.Label(self, text="FPS: --")
        self.fps_label.pack(pady=2)

        ttk.Label(self, text=f"Backend: {backend_name}").pack(pady=2)

    def update(self):
        self.fps_label.config(text=f"FPS: {self.state.fps:.1f}")


class ControlPanel(ttk.Frame):
    def __init__(self, parent, wh: WidthAndHeight, state: _SimulationState, backend_name: str = "Unknown"):
        super().__init__(parent)
        self.state = state

        self.simulation_panel = SimulationControlPanel(self, state)
        self.simulation_panel.pack(fill=tk.X, pady=(0, 10))

        self.camera_panel = CameraControlPanel(self, wh, state)
        self.camera_panel.pack(fill=tk.X, pady=(0, 10))

        self.info_panel = SimulationInfoPanel(self, state, backend_name)
        self.info_panel.pack(fill=tk.X, pady=(0, 10))

    def update(self):
        self.info_panel.update()
        self.simulation_panel.update()


class _SimulationFrame(tk.Frame):
    def __init__(self, parent, wh: WidthAndHeight, state: _SimulationState):
        super().__init__(parent)

        self.width = wh.RENDER_WIDTH
        self.height = wh.RENDER_HEIGHT

        self.state = state

        img = Image.fromarray(
            np.zeros_like(self.state.rgb_buffer, dtype=np.uint8)
        )
        self._cached_photo = ImageTk.PhotoImage(image=img)

        self.canvas = tk.Canvas(self, width=self.width, height=self.height, bg='black')
        self.canvas.create_image(self.width // 2, self.height // 2, image=self._cached_photo)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.buffer_update_timestamp = state.buffer_update_timestamp

    def display_image(self):
        try:
            current_timestamp = self.state.buffer_update_timestamp
            if current_timestamp and self.buffer_update_timestamp != current_timestamp:
                self.buffer_update_timestamp = current_timestamp

                self._cached_photo.paste(
                    Image.fromarray(self.state.rgb_buffer)
                )

                self.canvas.update_idletasks()
        except Exception as e:
            logging.getLogger(__name__).error(f"Display image error: {e}")
            self.canvas.delete("all")
            self.canvas.create_text(
                self.width // 2, self.height // 2,
                text=f"Display error:\n{str(e)}",
                fill="red", font=('Arial', 12)
            )

    def update(self):
        if self.state.has_recent_error:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.width // 2, self.height // 2,
                text=f"Error:\n{self.state.error_message}",
                fill="red", font=('Arial', 12), width=self.width - 20
            )
        else:
            self.display_image()


class _TopWindow(tk.Tk):
    def __init__(self, state: _SimulationState, wh: WidthAndHeight, backend_name: str = "Unknown"):
        super().__init__()

        self.state = state
        self.logger = logging.getLogger(__name__)

        self._setup_ui(wh, backend_name)
        self._schedule_ui_update()

    def _setup_ui(
            self, wh: WidthAndHeight, backend_name: str
    ):
        self.title("Generic Simulator Viewer")

        self.simulation_frame = _SimulationFrame(self, wh, self.state)
        self.simulation_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.control_panel = ControlPanel(self, wh, self.state, backend_name)
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)

    def _schedule_ui_update(self):
        try:
            self.simulation_frame.update()
            self.control_panel.update()
            self.after(UI_REFRESH_INTERVAL_MS, self._schedule_ui_update)
        except Exception as e:
            self.logger.error(f"UI update error: {e}")


class GenericTkinterViewer:
    def __init__(
            self,
            world_width: float, world_height: float,
            render_width: int, render_height: int,
            backend: SimulatorBackend
    ):
        wh = WidthAndHeight(world_width, world_height, render_width, render_height)

        self.backend = backend
        self.logger = logging.getLogger(__name__)

        self.state = _SimulationState(wh)

        self.simulation = _Simulation(backend, self.state)

        backend_name = getattr(backend, '__class__', type(backend)).__name__
        self._viewer = _TopWindow(self.state, wh, backend_name)

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def run(self):
        """Run the viewer with proper cleanup"""
        try:
            self.simulation.start()
            self._viewer.mainloop()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Runtime error: {e}")
        self.simulation.stop()
