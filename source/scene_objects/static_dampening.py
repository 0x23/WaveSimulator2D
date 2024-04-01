import wave_simulation
import cupy as cp


class StaticDampening(wave_simulation.SceneObject):
    """
    Implements a static dampening field that overwrites the entire domain.
    Therefore, us this as base layer in your scene.
    """

    def __init__(self, dampening_field, border_thickness):
        """
        Creates a static dampening field object
        @param dampening_field: The dampening field. If None, a default dampening field is applied.
        @param pml_thickness: Thickness of the Perfectly Matched Layer (PML) at the boundaries to prevent reflections.
        """
        w = dampening_field.shape[1]
        h = dampening_field.shape[0]
        self.d = cp.ones((h, w), dtype=cp.float32)
        self.d = cp.clip(cp.array(dampening_field), 0.0, 1.0)

        # apply border dampening
        for i in range(border_thickness):
            v = (i / border_thickness) ** 0.5
            self.d[i, i:w - i] = v
            self.d[-(1 + i), i:w - i] = v
            self.d[i:h - i, i] = v
            self.d[i:h - i, -(1 + i)] = v

    def render(self, wave_speed_field: cp.ndarray, dampening_field: cp.ndarray):
        assert (dampening_field.shape == self.d.shape)

        # overwrite existing dampening field
        dampening_field[:] = self.d

    def update_field(self, field: cp.ndarray, t):
        pass


