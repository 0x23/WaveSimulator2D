import wave_visualizer as vis
import wave_simulation as sim
import numpy as np
import cv2
import math


def load_scene_from_image(simulator, scene_image):
    # set refractive index field
    simulator.set_refractive_index_field(scene_image[:, :, 0]/100)

    # set absorber field
    simulator.set_dampening_field(1.0-scene_image[:, :, 2]/255, 48)

    # set sources
    sources_pos = np.flip(np.argwhere(scene_image[:, :, 1] > 0), axis=1)
    apf = np.tile(np.array([0, 1.0, 0.3]), (sources_pos.shape[0], 1))
    sources = np.concatenate((sources_pos, apf), axis=1)
    sources[:, 4] = scene_image[sources_pos[:, 1], sources_pos[:, 0], 1]/255*0.5  # set frequency to channel value

    simulator.set_sources(sources)


def main(scene_image_fn, num_iterations):
    scene_image = cv2.cvtColor(cv2.imread(scene_image_fn), cv2.COLOR_BGR2RGB)

    # create simulator and visualizer objects
    simulator = sim.WaveSimulator2D(scene_image.shape[1], scene_image.shape[0])
    visualizer = vis.WaveVisualizer(field_colormap=vis.get_colormap_lut('RdBu', True),
                                    intensity_colormap=vis.get_colormap_lut('afmhot', False, 0.1))

    # load scene from image file
    load_scene_from_image(simulator, scene_image)

    # create video writers
    video_writer1 = cv2.VideoWriter('simulation_field.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                   60, (scene_image.shape[1], scene_image.shape[0]))
    video_writer2 = cv2.VideoWriter('simulation_intensity.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                   60, (scene_image.shape[1], scene_image.shape[0]))

    # run simulation
    for i in range(num_iterations):
        simulator.update_sources()
        simulator.update_field()
        visualizer.update(simulator)

        if i % 2 == 0:
            frame_int = visualizer.render_intensity(1.25)
            frame_field = visualizer.render_field(0.7)
            cv2.imshow("Wave Simulation", frame_field)
            cv2.waitKey(1)
            video_writer1.write(frame_field)
            video_writer2.write(frame_int)


if __name__ == "__main__":
    main("scene_lens_doubleslit.png", 10000)
