import cupy
import numpy as np
import cupy as cp
import cupyx.scipy.signal
from abc import ABC, abstractmethod


class SceneObject(ABC):
    """
    Interface for simulation scene objects. A scene object is anything defining or modifying the simulation scene.
    For example: Light sources, Absorbers or regions with specific refractive index. Scene objects can change the
    simulated field and draw their contribution to the wave speed field and dampening field each frame """

    @abstractmethod
    def render(self, field: cupy.ndarray, wave_speed_field: cupy.ndarray, dampening_field: cupy.ndarray):
        """ renders the scene objects contribution to the wave speed field and dampening field """
        pass

    @abstractmethod
    def update_field(self, field: cupy.ndarray, t):
        """ performs updates to the field itself, e.g. for adding sources """
        pass

    @abstractmethod
    def render_visualization(self, image: np.ndarray):
        """ renders a visualization of the scene object to the image """
        pass


class WaveSimulator2D:
    """
    Simulates the 2D wave equation
    The system assumes units, where the wave speed is 1.0 pixel/timestep
    source frequency should be adjusted accordingly
    """
    def __init__(self, w, h, scene_objects, initial_field=None):
        """
        Initialize the 2D wave simulator.
        @param w: Width of the simulation grid.
        @param h: Height of the simulation grid.
        """
        self.global_dampening = 1.0
        self.c = cp.ones((h, w), dtype=cp.float32)                      # wave speed field (from refractive indices)
        self.d = cp.ones((h, w), dtype=cp.float32)                      # dampening field
        self.u = cp.zeros((h, w), dtype=cp.float32)                     # field values
        self.u_prev = cp.zeros((h, w), dtype=cp.float32)                # field values of prev frame

        if initial_field is not None:
            assert w == initial_field.shape[1] and h == initial_field.shape[2], 'width/height of initial field invalid'
            self.u[:] = initial_field
            self.u_prev[:] = initial_field

        # Define Laplacian kernel
        self.laplacian_kernel = cp.array([[0.066, 0.184, 0.066],
                                          [0.184, -1.0, 0.184],
                                          [0.066, 0.184, 0.066]])

        # self.laplacian_kernel = cp.array([[0.05, 0.2, 0.05],
        #                           [0.2, -1.0, 0.2],
        #                           [0.05, 0.2, 0.05]])

        # self.laplacian_kernel = cp.array([[0.103, 0.147, 0.103],
        #                                   [0.147, -1.0, 0.147],
        #                                   [0.103, 0.147, 0.103]])

        self.t = 0
        self.dt = 1.0

        self.scene_objects = scene_objects if scene_objects is not None else []

    def reset_time(self):
        """
        Reset the simulation time to zero.
        """
        self.t = 0.0

    def update_field(self):
        """
        Update the simulation field based on the wave equation.
        """
        # calculate laplacian using convolution
        laplacian = cupyx.scipy.signal.convolve2d(self.u, self.laplacian_kernel, mode='same', boundary='fill')

        # update field
        v = (self.u - self.u_prev) * self.d * self.global_dampening
        r = (self.u + v + laplacian * (self.c * self.dt)**2)

        self.u_prev[:] = self.u
        self.u[:] = r

        self.t += self.dt

    def update_scene(self):
        # clear wave speed field and dampening field
        self.c.fill(1.0)
        self.d.fill(1.0)

        for obj in self.scene_objects:
            obj.render(self.u, self.c, self.d)

        for obj in self.scene_objects:
            obj.update_field(self.u, self.t)

    def get_field(self):
        """
        Get the current state of the simulation field.
        @return: A 2D array representing the simulation field.
        """
        return self.u

    def render_visualization(self, image=None):
        # clear wave speed field and dampening field
        if image is None:
            image = np.zeros((self.c.shape[0], self.c.shape[1], 3), dtype=np.uint8)

        for obj in self.scene_objects:
            obj.render_visualization(image)

        return image


