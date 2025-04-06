from wave_sim2d.wave_simulation import SceneObject
import cupy as cp
import cupyx.scipy.signal
import numpy as np

class StrainRefractiveIndex(SceneObject):
    """
    Implements a dynamic refractive index field that linearly depends on the strain of the current field.
    The refractive index within the entire domain is overwritten
    """

    def __init__(self, refractive_index_offset, coupling_constant):
        """
        Creates a strain refractive index field object
        :param coupling_constant: coupling constant between the strain and the refractive index
        """
        self.coupling_constant = coupling_constant
        self.refractive_index_offset = refractive_index_offset

        self.du_dx_kernel = cp.array([[-1, 0.0, 1]])
        self.du_dy_kernel = cp.array([[-1], [0.0], [1]])

        self.strain_field = None

    def render(self, field: cp.ndarray, wave_speed_field: cp.ndarray, dampening_field: cp.ndarray):
        # compute strain
        du_dx = cupyx.scipy.signal.convolve2d(field, self.du_dx_kernel, mode='same', boundary='fill')
        du_dy = cupyx.scipy.signal.convolve2d(field, self.du_dy_kernel, mode='same', boundary='fill')

        self.strain_field = cp.sqrt(du_dx**2 + du_dy**2)

        # compute refractive index from strain
        refractive_index_field = self.refractive_index_offset + self.strain_field*self.coupling_constant

        # assign wave speed using refractive index from above
        wave_speed_field[:] = 1.0/cp.clip(cp.array(refractive_index_field), 0.9, 10.0)

    def update_field(self, field: cp.ndarray, t):
        pass

    def render_visualization(self, image: np.ndarray):
        """ renders a visualization of the scene object to the image """
        pass
