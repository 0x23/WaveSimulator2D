# 2D Wave Simulation on the GPU

This repository contains a lightweight 2D wave simulator running on the GPU using CuPy library (probably requires a NVIDIA GPU). It can be used for 2D light and sound simulations.
A simple visualizer shows the field and its intensity on the screen and writes a movie file for each to disks. The goal is to provide a fast, easy to use but still felxible wave simulator.

<div style="display: flex;">
    <img src="images/simulation_1.jpg" alt="Example Image 1" width="49%">
    <img src="images/simulation_2.jpg" alt="Example Image 2" width="49%">
</div>

### Update 01.04.2024

* Refactored the code to support a more flexible scene description. A simulation scene now consists of a list of objects that add their contribution to the fields.
They can be combined to build complex and time dependent simulations. The refactoring also made the core simulation code even simpler.
* Added a few new custom colormaps that work well for wave simulations.
* Added new examples, which should make it easier to understand the usage of the program and how you can setup your own simulations: [examples](source/examples).

<div style="display: flex;">
    <img src="images/simulation_3.jpg" alt="Example Image 3" width="45%">
    <img src="images/simulation_4.jpg" alt="Example Image 4" width="45%">
</div>

The old image based scene description is still available as a scene object. You can continue to use the convenience of an image editing software and create simulations
without much programming.

###  Image Scene Decsription Usage ###

When using the 'StaticImageScene' class the simulation scenes can given as an 8Bit RGB image with the following channel semantics:
* Red:   The Refractive index times 100 (for refractive index 1.5 you would use value 150)
* Green: Each pixel with a green value above 0 is a sinusoidal wave source. The green value defines its frequency.
* Blue:  Absorbtion field. Larger values correspond to higher dampening of the waves, use graduated transitions to avoid reflections

WARNING: Do not use anti-aliasing for the green channel ! The shades produced are interpreted as different source frequencies, which yields weird results.

<div style="display: flex;">
    <img src="images/source_antialiasing.png" alt="Example Image 5" width="50%">
</div>

### Recommended Installation ###

1. Go [here](https://github.com/conda-forge/miniforge) and install miniforge/mamba, which is a python package manager.
2. Start the mamba command prompt (under windows type 'Miniforge Prompt' in the start menu and you should find it).
3. install the dependencies by running:
   - **mamba install numpy, opencv, matplotlib, cupy**
4. Run the program directly from the miniforge prompt (cd into the directory where you downloaded the Wave Simulator first):
   python main.py
6. Alternatively, you can run the program from an IDE like PyCharm (don't forget to configure the IDE to use the python interpreter from the mamba/miniforge install directory)





