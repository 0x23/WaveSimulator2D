import wave_visualizer as vis
import wave_simulation as sim
import numpy as np
import cupy as cp
import math
import cv2
from scene_objects.static_dampening import StaticDampening
from scene_objects.static_refractive_index import StaticRefractiveIndex
from scene_objects.source import PointSource


def gaussian_kernel(size, sigma):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


class MovingCharge(sim.SceneObject):
    """
    Implements a point source scene object. The amplitude can be optionally modulated using a modulator object.
    :param x: center position x.
    :param y: center position y.
    :param frequency: motion frequency
    :param amplitude: motion amplitude
    """
    def __init__(self, x, y, frequency, amplitude):
        self.x = x
        self.y = y
        self.frequency = frequency
        self.amplitude = amplitude
        self.size = 11

        # create a smooth source shape
        self.source_array = cp.array(gaussian_kernel(self.size, self.size/3))

    def render(self, wave_speed_field, dampening_field):
        # no changes to the refractive index or dampening field required for this class
        pass

    def update_field(self, field, t):
        fade_in = math.sin(min(t*0.1, math.pi/2))

        # write the moving charge to the field
        x = self.x + math.sin(self.frequency * t*0.05)*200
        y = self.y + math.sin(self.frequency * t)*self.amplitude

        # copy source shape to current position into field
        wh = self.source_array.shape[1]//2
        hh = self.source_array.shape[0]//2
        field[y-hh:y+hh+1, x-wh:x+wh+1] += self.source_array * fade_in * 0.25


def build_scene():
    """
    In this example, a custom scene object is implemented and used to simulate a moving field disturbance.
    """
    width = 600
    height = 600
    objects = []

    # Add a static dampening field without any dampending in the interior (value 1.0 means no dampening)
    # However a dampening layer at the border is added to avoid reflections (see parameter 'border thickness')
    objects.append(StaticDampening(np.ones((height, width)), 64))

    # add a constant refractive index field
    objects.append(StaticRefractiveIndex(np.full((height, width), 1.5)))

    # add a simple point source
    objects.append(MovingCharge(300, 300, 0.1, 10))

    return objects, width, height


def main():
    # create colormaps
    field_colormap = vis.get_colormap_lut('colormap_wave1', invert=False, black_level=-0.05)
    intensity_colormap = vis.get_colormap_lut('afmhot', invert=False, black_level=0.0)

    # reset random number generator
    np.random.seed(0)

    # build simulation scene
    scene_objects, w, h = build_scene()

    # create simulator and visualizer objects
    simulator = sim.WaveSimulator2D(w, h, scene_objects)
    visualizer = vis.WaveVisualizer(field_colormap=field_colormap, intensity_colormap=intensity_colormap)

    # run simulation
    for i in range(8000):
        simulator.update_scene()
        simulator.update_field()
        visualizer.update(simulator)

        # visualize very N frames
        if (i % 2) == 0:
            # show field
            frame_field = visualizer.render_field(1.0)
            cv2.imshow("Wave Simulation Field", frame_field)

            # show intensity
            # frame_int = visualizer.render_intensity(1.0)
            # cv2.imshow("Wave Simulation Intensity", frame_int)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()

