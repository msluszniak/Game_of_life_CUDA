#pragma once
#include <cassert>
#include <iostream>
#include <fstream>
#include <Windows.h>
#include <stdio.h>
#include <helper_gl.h>
#include <helper_cuda.h>
#include <cuda_gl_interop.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <GL/freeglut.h>
#include <GLFW/glfw3.h>

#define WHITE_COLOR -1
#define BLACK_COLOR 0

struct inputer_data {
    int color;
    int background_color;
    u_int num_threads;
    u_int image_width;
    u_int image_height;
    u_int total_size;
    bool* host_map;
    bool* host_next_state;
};

char get_mode();

void input_file(inputer_data* input);

void input_manual(inputer_data* input);

inputer_data input_wrapper();