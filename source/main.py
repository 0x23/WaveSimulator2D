import wave_visualizer as vis
import wave_simulation as sim
import numpy as np
import cv2
import math
import json


def load_scene_from_image(simulator, scene_image, source_brightness_scale=1.0):
    """
    load source from an image description
    The simulation scenes are given as an 8Bit RGB image with the following channel semantics:
        * Red:   The Refractive index times 100 (for refractive index 1.5 you would use value 150)
        * Green: Each pixel with a green value above 0 is a sinusoidal wave source. The green value defines its frequency.
                 WARNING: Do not use anti aliasing for the green channel !
        * Blue:  Absorbtion field. Larger values correspond to higher dampening of the waves, use graduated transitions to avoid reflections
    """
    # set refractive index field
    simulator.set_refractive_index_field(scene_image[:, :, 0]/100)

    # set absorber field
    simulator.set_dampening_field(1.0-scene_image[:, :, 2]/255, 48)

    # set sources
    sources_pos = np.flip(np.argwhere(scene_image[:, :, 1] > 0), axis=1)
    phase_amplitude_freq = np.tile(np.array([0, 1.0, 0.3]), (sources_pos.shape[0], 1))
    sources = np.concatenate((sources_pos, phase_amplitude_freq), axis=1)

    sources[:, 4] = scene_image[sources_pos[:, 1], sources_pos[:, 0], 1]/255*0.5  # set frequency to channel value

    simulator.set_sources(sources)


def load_source_from_json(filename):
    """
    loads sources from json, this allows for better control over frequency, amplitude and phase
    and avoids aliasing problems
    """
    # TODO: implement me

def main(scene_image_fn, num_iterations, simulation_steps_per_frame, write_videos):
    scene_image = cv2.cvtColor(cv2.imread(scene_image_fn), cv2.COLOR_BGR2RGB)

    # create simulator and visualizer objects
    # good colormaps for field: RdBu[invert=True], colormap_wave1, colormap_wave2, icefire
    simulator = sim.WaveSimulator2D(scene_image.shape[1], scene_image.shape[0])
    visualizer = vis.WaveVisualizer(field_colormap=vis.get_colormap_lut('RdBu', invert=True),
                                    intensity_colormap=vis.get_colormap_lut('afmhot', invert=False, black_level=0.1))

    # load scene from image file
    load_scene_from_image(simulator, scene_image)

    # create video writers
    if write_videos:
        video_writer1 = cv2.VideoWriter('simulation_field.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                       60, (scene_image.shape[1], scene_image.shape[0]))
        video_writer2 = cv2.VideoWriter('simulation_intensity.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                       60, (scene_image.shape[1], scene_image.shape[0]))

    # run simulation
    for i in range(num_iterations):
        simulator.update_sources()
        simulator.update_field()
        visualizer.update(simulator)

        if i % simulation_steps_per_frame == 0:
            frame_int = visualizer.render_intensity(1.0)
            frame_field = visualizer.render_field(0.7)

           # frame_int = cv2.pyrDown(frame_int)
           # frame_field = cv2.pyrDown(frame_field)
            cv2.imshow("Wave Simulation", frame_field) #cv2.resize(frame_int, dsize=(1024, 1024)))
            cv2.waitKey(1)

            if write_videos:
                video_writer1.write(frame_field)
                video_writer2.write(frame_int)


if __name__ == "__main__":
    # increase simulation_steps_per_frame to better utilize GPU
    main("../example_scenes/scene_lens_doubleslit.png", 10000, simulation_steps_per_frame=4, write_videos=False)
