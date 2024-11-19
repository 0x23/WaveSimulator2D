import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))  # noqa

import numpy as np
import cv2
import wave_sim2d.wave_visualizer as vis
import wave_sim2d.wave_simulation as sim
from wave_sim2d.scene_objects.static_dampening import StaticDampening
from wave_sim2d.scene_objects.static_refractive_index import StaticRefractiveIndex
from wave_sim2d.scene_objects.source import PointSource, ModulatorSmoothSquare


def build_scene():
    """
    In this example, a new scene is created from scratch and a few emitters are places manually.
    One of the emitters uses an amplitude modulation object to change brightness over time
    """
    width = 600
    height = 600
    objects = []

    # Add a static dampening field without any dampending in the interior (value 1.0 means no dampening)
    # However a dampening layer at the border is added to avoid reflections (see parameter 'border thickness')
    objects.append(StaticDampening(np.ones((height, width)), 32))

    # add a constant refractive index field
    objects.append(StaticRefractiveIndex(np.full((height, width), 1.5)))

    # add a simple point source
    objects.append(PointSource(200, 220, 0.2, 8))

    # add a point source with an amplitude modulator
    amplitude_modulator = ModulatorSmoothSquare(0.025, 0.0, smoothness=0.5)
    objects.append(PointSource(200, 380, 0.2, 8, amp_modulator=amplitude_modulator))

    return objects, width, height


def main():
    # create colormaps
    field_colormap = vis.get_colormap_lut('colormap_wave4', invert=False, black_level=-0.05)
    intensity_colormap = vis.get_colormap_lut('afmhot', invert=False, black_level=0.0)

    # reset random number generator
    np.random.seed(0)

    # build simulation scene
    scene_objects, w, h = build_scene()

    # create simulator and visualizer objects
    simulator = sim.WaveSimulator2D(w, h, scene_objects)
    visualizer = vis.WaveVisualizer(field_colormap=field_colormap, intensity_colormap=intensity_colormap)

    # run simulation
    for i in range(2000):
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

