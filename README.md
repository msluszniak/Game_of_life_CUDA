# Game_of_life_CUDA
Classic game of life calculated on GPU (Cuda) and Ploting (OpenGL) on Windows


# Before you start...
To build and run project correctly you need to install CUDA. One of the easiest way to do it is to install Visual Studio (https://visualstudio.microsoft.com/downloads/).
When you succesfully install VS, then install CUDA from this link https://developer.nvidia.com/cuda-downloads. CUDA will be automatically configured during installation.
After that you need to install OpenGL. My tip: at the end of CUDA installation you may tick "cuda samples" box. Then OpenGL libraries will be in directory.


# Project
Program computes states of cellular automaton using CUDA and plot then state on the screen. You might choose whether you what to input data manually or from text file.
Also you can change colors of visualization.

To control speed of simulation use keyboard buttons as follow:

Key | Action
--- | ---
<kbd>S</kbd> | slow down simulation
<kbd>F</kbd> | speed up simlulation
<kbd>D</kbd> | go back to default value



# Samples

Glider gun <br> <br>
![glider_gun](glider_gun.gif "Glider gun")

Random initial positions <br> <br>
![random](random.gif "Random init positions")

