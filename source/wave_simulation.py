import numpy as np
import cupy as cp
import cupyx.scipy.signal


class WaveSimulator2D:
    """
    Simulates the 2D wave equation
    The system assumes units, where the wave speed is 1.0 pixel/timestep
    source frequency should be adjusted accordingly
    """
    def __init__(self, w, h):
        """
        Initialize the 2D wave simulator.
        @param w: Width of the simulation grid.
        @param h: Height of the simulation grid.
        """
        self.global_dampening = 1.0
        self.source_opacity = 0.9       # opacity of source pixels to incoming waves. If the opacity is 0.0
                                        # the field will be completely over written by the source term
                                        # a nonzero value (e.g 0.5) allows for antialiasing of sources to work

        self.c = cp.ones((h, w), dtype=cp.float32)                      # wave speed field (from refractive indices)
        self.d = cp.ones((h, w), dtype=cp.float32)                      # dampening field
        self.u = cp.zeros((h, w), dtype=cp.float32)                     # field values
        self.u_prev = cp.zeros((h, w), dtype=cp.float32)                # field values of prev frame

        # set boundary dampening to prevent reflections
        self.set_dampening_field(None, 32)

        # Define Laplacian kernel
        self.laplacian_kernel = cp.array([[0.05, 0.2, 0.05],
                                          [0.2, -1.0, 0.2],
                                          [0.05, 0.2, 0.05]])

        # self.laplacian_kernel = cp.array([[0.103, 0.147, 0.103],
        #                                   [0.147, -1.0, 0.147],
        #                                   [0.103, 0.147, 0.103]])

        self.t = 0
        self.dt = 1.0
        self.sources = cp.zeros([0, 5])

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
        v = (self.u - self.u_prev) * self.d
        r = (self.u + v + laplacian * (self.c * self.dt)**2)

        self.u_prev[:] = self.u
        self.u[:] = r

        self.t += self.dt

    def get_field(self):
        """
        Get the current state of the simulation field.
        @return: A 2D array representing the simulation field.
        """
        return self.u

    def set_dampening_field(self, d, pml_thickness):
        """
        Set the dampening field for the simulation, which can be used to prevent reflections at boundaries.
        @param d: The dampening field. If None, a default dampening field is applied.
        @param pml_thickness: Thickness of the Perfectly Matched Layer (PML) at the boundaries to prevent reflections.
        """
        if d is not None:
            assert(d.shape == self.d.shape)
            self.d = cp.clip(cp.array(d), 0.0, self.global_dampening)
        else:
            self.d[:] = self.global_dampening

        w = self.d.shape[1]
        h = self.d.shape[0]
        for i in range(pml_thickness):
            v = (i / pml_thickness) ** 0.5
            self.d[i, i:w - i] = v
            self.d[-(1 + i), i:w - i] = v
            self.d[i:h - i, i] = v
            self.d[i:h - i, -(1 + i)] = v

    def set_refractive_index_field(self, r):
        """
        Set the refractive index field, which affects the wave speed in the simulation.
        @param r: The refractive index field.
        """
        assert(r.shape == self.c.shape)
        self.c = 1.0/cp.clip(cp.array(r), 1.0, 10.0)

    def set_sources(self, sources):
        """
        Set sources for the simulation.
        @param sources: An array of sources, where each source consists of 5 values: x, y, phase, amplitude, frequency.
        """
        assert sources.shape[1] == 5, 'sources must have shape Nx5'
        self.sources = cp.array(sources).astype(cp.float32)

    def update_sources(self):
        """
        Update the sources in the simulation field based on their properties.
        """
        v = cp.sin(self.sources[:, 2]+self.sources[:, 4]*self.t)*self.sources[:, 3]
        coords = self.sources[:, 0:2].astype(cp.int32)

        t = self.source_opacity
        self.u[coords[:, 1], coords[:, 0]] = self.u[coords[:, 1], coords[:, 0]]*t + v*(1.0-t)

