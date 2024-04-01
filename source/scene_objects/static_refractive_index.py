import wave_simulation
import cupy as cp


class StaticRefractiveIndex(wave_simulation.SceneObject):
    """
    Implements a static refractive index field that overwrites the entire domain.
    Therefore, us this as base layer in your scene.
    """

    def __init__(self, refractive_index_field):
        """
        Creates a static refractive index field object
        @param refractive_index_field: The refractive index field.
        """
        shape = refractive_index_field.shape
        self.c = cp.ones((shape[0], shape[1]), dtype=cp.float32)
        self.c = 1.0/cp.clip(cp.array(refractive_index_field), 0.9, 10.0)

    def render(self, wave_speed_field: cp.ndarray, dampening_field: cp.ndarray):
        assert (wave_speed_field.shape == self.c.shape)
        wave_speed_field[:] = self.c

    def update_field(self, field: cp.ndarray, t):
        pass


