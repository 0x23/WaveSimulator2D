import wave_visualizer
import wave_visualizer as vis
import wave_simulation as sim
import numpy as np
import cv2
import math
import json
from scene_objects.static_dampening import StaticDampening
from scene_objects.static_refractive_index import StaticRefractiveIndex
from scene_objects.static_image_scene import StaticImageScene
from scene_objects.source import PointSource, ModulatorSmoothSquare, ModulatorDiscreteSignal


def build_example_scene1(scene_image):
    """
    This example uses the old image scene description. See 'StaticImageScene' for more information.
    """
    scene_objects = [StaticImageScene(scene_image)]
    return scene_objects


def build_example_scene2(width, height):
    """
    In this example, a new scene is created from scratch and a few emitters are places manually.
    One of the emitters uses an amplitude modulation object to change brightness over time
    """
    objects = []

    # Add a static dampening field without any dampending in the interior (value 1.0 means no dampening)
    # However a dampening layer at the border is added to avoid reflections (see parameter 'border thickness')
    objects.append(StaticDampening(np.ones((height, width)), 48))

    # add a constant refractive index field
    objects.append(StaticRefractiveIndex(np.full((height, width), 1.5)))

    # add a simple point source
    objects.append(PointSource(200, 250, 0.19, 5))

    # add a point source with an amplitude modulator
    amplitude_modulator = ModulatorDiscreteSignal(np.random.randint(2, size=64), 0.0006)
    objects.append(PointSource(200, 350, 0.19, 5, amp_modulator=amplitude_modulator))

    return objects


def simulate(scene_image_fn, num_iterations,
             simulation_steps_per_frame, write_videos,
             field_colormap, intensity_colormap,
             background_image_fn=None):
    # reset random number generator
    np.random.seed(0)

    # load scene image
    scene_image = cv2.cvtColor(cv2.imread(scene_image_fn), cv2.COLOR_BGR2RGB)

    background_image = None
    if background_image_fn is not None:
        background_image = cv2.imread(background_image_fn)
        background_image = cv2.resize(background_image, (scene_image.shape[1], scene_image.shape[0]))

    # create simulator and visualizer objects
    simulator = sim.WaveSimulator2D(scene_image.shape[1], scene_image.shape[0])
    visualizer = vis.WaveVisualizer(field_colormap=field_colormap, intensity_colormap=intensity_colormap)

    # build simulation scene
    simulator.scene_objects = build_example_scene2(scene_image.shape[1], scene_image.shape[0])

    # create video writers
    if write_videos:
        video_writer1 = cv2.VideoWriter('simulation_field.avi', cv2.VideoWriter_fourcc(*'FFV1'),
                                       60, (scene_image.shape[1], scene_image.shape[0]))
        video_writer2 = cv2.VideoWriter('simulation_intensity.avi', cv2.VideoWriter_fourcc(*'FFV1'),
                                       60, (scene_image.shape[1], scene_image.shape[0]))

    # run simulation
    for i in range(num_iterations):
        simulator.update_scene()
        simulator.update_field()
        visualizer.update(simulator)

        if i % simulation_steps_per_frame == 0:
            frame_int = visualizer.render_intensity(1.0)
            frame_field = visualizer.render_field(1.0)

            if background_image is not None:
                frame_int = cv2.add(background_image, frame_int)
                frame_field = cv2.add(background_image, frame_field)

           # frame_int = cv2.pyrDown(frame_int)
           # frame_field = cv2.pyrDown(frame_field)
            cv2.imshow("Wave Simulation", frame_field) #cv2.resize(frame_int, dsize=(1024, 1024)))
            cv2.waitKey(1)

            if write_videos:
                video_writer1.write(frame_field)
                video_writer2.write(frame_int)

        if i % 128 == 0:
            print(f'{int((i+1)/num_iterations*100)}%')


if __name__ == "__main__":
    print('This file contains tests for development and you may not bve able to run it without errors')
    print('Please take a look at the previded examples')

    # increase simulation_steps_per_frame to better utilize GPU
    # good colormaps for field: RdBu[invert=True], colormap_wave1, colormap_wave2, colormap_wave4, icefire
    simulate('../exxample_data/scene_lens_doubleslit.png',
             20000,
             simulation_steps_per_frame=16,
             write_videos=True,
             field_colormap=vis.get_colormap_lut('colormap_wave4', invert=False, black_level=-0.05),
#             field_colormap=vis.get_colormap_lut('RdBu', invert=True, make_symmetric=True),
             intensity_colormap=vis.get_colormap_lut('afmhot', invert=False, black_level=0.0),
             background_image_fn=None)

