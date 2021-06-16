
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GLUT_DISABLE_ATEXIT_HACK
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
#include "GUI.h"
#include "inputer.h"
#include "window_handler.h"

#define DEFAULT_DELAY 50
#define INCREASE_BY 1 / 1.1
#define DECREASE_BY 1.1
#define WHITE_COLOR -1
#define BLACK_COLOR 0


constexpr u_int max_num_blocks = 32768;

using u_int = unsigned int;
using u_short = unsigned short;

u_int delay = DEFAULT_DELAY;

__global__ void next_step(bool* map, bool* next_state, inputer_data input, int32_t* image ) {

    for (u_int idx = (u_int)(blockIdx.x * blockDim.x + threadIdx.x); idx < input.total_size; idx += (u_int)(blockDim.x * gridDim.x)) {
         //now we need to organize 2d in 1d array
         //for each element in map array we want to calculate number of alive neighbours
         //so we need to figure out how to indexing through given array
         //second problem is that we need to compute this but with respect of "curved borders" of the map

        //x positions of actual idx and right and left neighbour
        u_int x = idx % input.image_width;
        u_int x_left = (x + input.image_width - 1) % input.image_width;
        u_int x_right = (x + 1) % input.image_width;

        //y coords analogically as x
        u_int y = idx - x;
        u_int y_top = (y + input.total_size - input.image_width) % input.total_size;
        u_int y_bottom = (y + input.image_width) % input.total_size;

        u_short num_neigbhbour_alive = 0;

        //top neighbours
        num_neigbhbour_alive += (u_short)map[x_left + y_top] + (u_short)map[x + y_top] + (u_short)map[x_right + y_top];
        //left & right neighbours
        num_neigbhbour_alive += (u_short)map[x_left + y] + (u_short)map[x_right + y];
        //bottom neighbours
        num_neigbhbour_alive += (u_short)map[x_left + y_bottom] + (u_short)map[x + y_bottom] + (u_short)map[x_right + y_bottom];

        //compute state and set color
        next_state[x + y] = num_neigbhbour_alive == 3 || (num_neigbhbour_alive == 2 && map[x + y]);

        //this might be extended to all color palette
        //image[x + y] = next_state[x + y] ? WHITE_COLOR : BLACK_COLOR;
        image[x + y] = next_state[x + y] ? input.color : input.background_color;
    }
}

__host__ void calculate_map(bool*& map, bool*& next_state, inputer_data input, int32_t*&out_data) {
    assert(input.image_width * input.image_height % input.num_threads == 0);
    u_int requested_blocks = ((input.image_width * input.image_height) / input.num_threads);
    u_int num_blocks = (u_int)min(max_num_blocks, requested_blocks);
    next_step <<< num_blocks, input.num_threads >>> (map, next_state, input, out_data);
    std::swap(map, next_state);
}

// function generates 
__host__ void generate_cuda_image(Window_handler *window_handler) {
    int32_t* out_data;

    out_data = reinterpret_cast<int32_t*>(window_handler->get_cuda_dest_resource());

    //calculate grid size
    dim3 block(16, 16, 1);
    dim3 grid(window_handler->get_input().image_width / block.x, window_handler->get_input().image_height / block.y, 1);

    // compute kernel function
    calculate_map(window_handler->map, window_handler->next_state, window_handler->get_input(), out_data);

    // now handle texture on device (GPU)
    cudaArray* texture_ptr;
    cudaGraphicsMapResources(1, &(window_handler->cuda_tex_result_resource), 0);
    cudaGraphicsSubResourceGetMappedArray(&texture_ptr, window_handler->cuda_tex_result_resource, 0, 0);

    int num_texels = window_handler->get_input().image_width * window_handler->get_input().image_height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    cudaMemcpyToArray(texture_ptr, 0, 0, window_handler->cuda_dest_resource, size_tex_data, cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &(window_handler->cuda_tex_result_resource), 0);
}

// wrapper for displaying image
void display(Window_handler &window_handler){
    generate_cuda_image(&window_handler);
    displayImage(window_handler.get_tex_cudaResult(), window_handler.get_shDrawTex());
    cudaDeviceSynchronize();
}

void error_callback(int error, const char* description){
    fprintf(stderr, "Error: %s\n", description);
}

//helper functions for key_callback
void speed_up() {
    delay = (int)delay * INCREASE_BY;
}

void slow_down() {
    delay = (int)delay * DECREASE_BY;
}

void default_delay() {
    delay = DEFAULT_DELAY;
}

//function which speed up or slow down paste of displaying or reset to default
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){
    if (key == GLFW_KEY_F && action == GLFW_PRESS)
        speed_up();
    else if (key == GLFW_KEY_S && action == GLFW_PRESS)
        slow_down();
    else if (key == GLFW_KEY_D && action == GLFW_PRESS)
        default_delay();
}

int main(int argc, char** argv) {
    struct inputer_data input = input_wrapper();
    Window_handler window_handler = Window_handler(input);
    window_handler.start();
    window_handler.preprocess_window();
    glfwSetErrorCallback(error_callback);
    while (!glfwWindowShouldClose(window_handler.get_window())) {
        display(window_handler);
        glfwSetKeyCallback(window_handler.get_window(), key_callback);
        window_handler.handle_window();
        Sleep(delay);
    }
    window_handler.clean_window();
    return 0;
}


