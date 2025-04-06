import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))  # noqa

import cv2
import numpy as np
import cupy as cp
import wave_sim2d.wave_visualizer as vis
import wave_sim2d.wave_simulation as sim
from wave_sim2d.scene_objects.source import *
from wave_sim2d.scene_objects.static_refractive_index import *
from wave_sim2d.scene_objects.static_dampening import *


def build_scene():
    """
    This example creates fabry pirot cavity and shows the standing waves
    """
    width = 768
    height = 512
    objects = []

    # Add a static dampening field without any dampening in the interior (value 1.0 means no dampening)
    # However a dampening layer at the border is added to avoid reflections (see parameter 'border thickness')
    objects.append(StaticDampening(np.ones((height, width)), 48))

    # add nonlinear refractive index field
    objects.append(StaticRefractiveIndexBox((50, height//2), (50, int(height*0.8)), 0.0, 100.0))
    objects.append(StaticRefractiveIndexBox((width-180, height//2), (40, int(height*0.8)), 0.0, 10.0))

    # add a point source with an amplitude modulator
    objects.append(LineSource((80, height//2-100), (80, height//2+100), 0.0395, 0.4))

    return objects, width, height


def show_field(field, brightness_scale):
    gray = (cp.clip(field*brightness_scale, -1.0, 1.0) * 127 + 127).astype(np.uint8)
    img = gray.get()
    cv2.imshow("Strain Simulation Field", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    # create colormaps
    field_colormap = vis.get_colormap_lut('colormap_wave1', invert=False, black_level=-0.05)
    intensity_colormap = vis.get_colormap_lut('afmhot', invert=False, black_level=0.0)

    # build simulation scene
    scene_objects, w, h = build_scene()

    # create simulator and visualizer objects
    simulator = sim.WaveSimulator2D(w, h, scene_objects)
    visualizer = vis.WaveVisualizer(field_colormap=field_colormap, intensity_colormap=intensity_colormap)

    # run simulation
    for i in range(100000):
        simulator.update_scene()
        simulator.update_field()
        visualizer.update(simulator)

        # show field
        frame_field = visualizer.render_field(1.0)
        cv2.imshow("Wave Simulation Field", frame_field)

        # show intensity
        # frame_int = visualizer.render_intensity(1.0)
        # cv2.imshow("Wave Simulation Intensity", frame_int)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()

