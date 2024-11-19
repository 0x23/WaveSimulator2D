from wave_sim2d.wave_simulation import SceneObject
import cupy as cp
import cupyx.scipy.signal


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

        self.strain_kernel = cp.array([[0.05, 0.2, 0.05],
                                       [0.2, -1.0, 0.2],
                                       [0.05, 0.2, 0.05]])

        self.strain_field = None

    def render(self, field: cp.ndarray, wave_speed_field: cp.ndarray, dampening_field: cp.ndarray):
        # compute strain
        self.strain_field = cupyx.scipy.signal.convolve2d(field, self.strain_kernel, mode='same', boundary='fill')

        # compute refractive index from strain
        refractive_index_field = self.refractive_index_offset + self.strain_field*self.coupling_constant

        # assign wave speed using refractive index from above
        wave_speed_field[:] = 1.0/cp.clip(cp.array(refractive_index_field), 0.9, 10.0)

    def update_field(self, field: cp.ndarray, t):
        pass


