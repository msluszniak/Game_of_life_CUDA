
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

constexpr u_int max_num_blocks = 32768;

using u_int = unsigned int;
using u_short = unsigned short;

u_int delay = 250;


__global__ void next_step(bool* map, bool* next_state, u_int width, u_int length, int32_t* image ) {
    u_int total_size = width * length;

    for (u_int idx = (u_int)(blockIdx.x * blockDim.x + threadIdx.x); idx < total_size; idx += (u_int)(blockDim.x * gridDim.x)) {
         //now we need to organize 2d in 1d array
         //for each element in map array we want to calculate number of alive neighbours
         //so we need to figure out how to indexing through given array
         //second problem is that we need to compute this but with respect of "curved borders" of the map

        //x positions of actual idx and right and left neighbour
        u_int x = idx % width;
        u_int x_left = (x + width - 1) % width;
        u_int x_right = (x + 1) % width;

        //y coords analogically as x
        u_int y = idx - x;
        u_int y_top = (y + total_size - width) % total_size;
        u_int y_bottom = (y + width) % total_size;

        u_short num_neigbhbour_alive = 0;

        //top neighbours
        num_neigbhbour_alive += (u_short)map[x_left + y_top] + (u_short)map[x + y_top] + (u_short)map[x_right + y_top];
        //left & right neighbours
        num_neigbhbour_alive += (u_short)map[x_left + y] + (u_short)map[x_right + y];
        //bottom neighbours
        num_neigbhbour_alive += (u_short)map[x_left + y_bottom] + (u_short)map[x + y_bottom] + (u_short)map[x_right + y_bottom];

        //compute state and set color
        next_state[x + y] = num_neigbhbour_alive == 3 || (num_neigbhbour_alive == 2 && map[x + y]);
        image[x + y] = next_state[x + y] ? -1 : 0;
    }
}


__host__ void calculate_map(bool*& map, bool*& next_state, u_int image_width, u_int image_height, u_short num_threads, int32_t*&out_data) {
    assert(image_width * image_height % num_threads == 0);
    u_int requested_blocks = ((image_width * image_height) / num_threads);
    u_int num_blocks = (u_int)min(max_num_blocks, requested_blocks);
    next_step <<< num_blocks, num_threads >>> (map, next_state, image_width, image_height, out_data);
    std::swap(map, next_state);
}



//copy image and process using CUDA
//void generateCUDAImage(u_int image_width, u_int image_height, bool*& map, bool*& next_state, u_int num_threads, cudaGraphicsResource* cuda_dest_resource, 
//    cudaGraphicsResource* cuda_tex_result_resource){
//    //run the Cuda kernel
//    int32_t* out_data;
//
//    out_data = reinterpret_cast<int32_t*>(cuda_dest_resource);
//    //calculate grid size
//    dim3 block(16, 16, 1);
//    dim3 grid(image_width / block.x, image_height / block.y, 1);
//    calculate_map(map, next_state, image_width, image_height, num_threads, out_data);
//
//    cudaArray* texture_ptr;
//    cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0);
//    cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0);
//
//    int num_texels = image_width * image_height;
//    int num_values = num_texels * 4;
//    int size_tex_data = sizeof(GLubyte) * num_values;
//    cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource, size_tex_data, cudaMemcpyDeviceToDevice);
//
//    cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0);
//}

//copy image and process using CUDA
void generateCUDAImage(Window_handler *window_handler) {
    //run the Cuda kernel
    int32_t* out_data;

    out_data = reinterpret_cast<int32_t*>(window_handler->get_cuda_dest_resource());
    //calculate grid size
    dim3 block(16, 16, 1);
    dim3 grid(window_handler->get_input().image_width / block.x, window_handler->get_input().image_height / block.y, 1);
    calculate_map(window_handler->map, window_handler->next_state, window_handler->get_input().image_width, window_handler->get_input().image_height, window_handler->get_input().num_threads, out_data);

    cudaArray* texture_ptr;
    cudaGraphicsMapResources(1, &(window_handler->cuda_tex_result_resource), 0);
    cudaGraphicsSubResourceGetMappedArray(&texture_ptr, window_handler->cuda_tex_result_resource, 0, 0);

    int num_texels = window_handler->get_input().image_width * window_handler->get_input().image_height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    cudaMemcpyToArray(texture_ptr, 0, 0, window_handler->cuda_dest_resource, size_tex_data, cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &(window_handler->cuda_tex_result_resource), 0);
}


//void display(u_int image_width, u_int image_height, bool*& map, bool*& next_state, u_int num_threads, GLint shDrawTex, GLuint tex_cudaResult, 
//    cudaGraphicsResource* cuda_dest_resource, cudaGraphicsResource* cuda_tex_result_resource){
//    generateCUDAImage(image_width, image_height, map, next_state, num_threads, cuda_dest_resource, cuda_tex_result_resource);
//    displayImage(tex_cudaResult, shDrawTex);
//    cudaDeviceSynchronize();
//}

void display(Window_handler &window_handler){
    //generateCUDAImage(image_width, image_height, map, next_state, num_threads, cuda_dest_resource, cuda_tex_result_resource);
    generateCUDAImage(&window_handler);
    displayImage(window_handler.get_tex_cudaResult(), window_handler.get_shDrawTex());
    cudaDeviceSynchronize();
}


void error_callback(int error, const char* description){
    fprintf(stderr, "Error: %s\n", description);
}

void speed_up() {
    delay = (int)delay * 0.9;
}

void slow_down() {
    delay = (int)delay * 1 / (0.9);
}

void default_delay() {
    delay = 250;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){
    if (key == GLFW_KEY_F && action == GLFW_PRESS)
        speed_up();
    else if (key == GLFW_KEY_S && action == GLFW_PRESS)
        slow_down();
    else if (key == GLFW_KEY_D && action == GLFW_PRESS)
        default_delay();
}


int main(int argc, char** argv) {
    //u_int image_width, image_height, total_size, num_threads;


    //bool *map, *next_state;


    //bool *host_map, *host_next_state;
    //bool* result;


    //struct cudaGraphicsResource* cuda_dest_resource;
    //struct cudaGraphicsResource* cuda_tex_result_resource = NULL;
    struct inputer_data input = input_wrapper();
    Window_handler window_handler = Window_handler(input);
    //inputer(num_threads, image_width, image_height, total_size, host_map, host_next_state);

    //cudaMalloc((void**)&map, sizeof(bool) * input.total_size);
    //cudaMalloc((void**)&next_state, sizeof(bool) * input.total_size);
    //cudaMalloc((void**)&cuda_dest_resource, sizeof(int32_t) * input.total_size);
    //cudaMemset(map, false, input.total_size);
    //cudaMemset(next_state, false, input.total_size);
    //std::cout << "Do you want to start [Y/n]?" << std::endl;
    //char decision;
    //std::cin >> decision;
    //if (decision != 'y' && decision != 'Y') {
    //    std::cout << "Ending" << std::endl;
    //    return 0;
    //}

    //glfwSetErrorCallback(error_callback);
    //glfwInit();
    //GLFWwindow* window = glfwCreateWindow(512, 512, "Okienko", NULL, NULL);
    ////GLFWwindow* window = glfwCreateWindow(800, 800, "Okienko", NULL, NULL);
    //glfwMakeContextCurrent(window);
    //glClearColor(0.5, 0.5, 0.5, 1.0);
    //glDisable(GL_DEPTH_TEST);

    //GLuint tex_cudaResult;

    //cudaMemcpy(map, input.host_map, sizeof(bool) * input.total_size, cudaMemcpyHostToDevice);
    //cudaMemcpy(next_state, input.host_next_state, sizeof(bool) * input.total_size, cudaMemcpyHostToDevice);
    //createTextureDst(&tex_cudaResult, input.image_width, input.image_height, cuda_tex_result_resource);

    //glewInit();
    ////GLint shDraw = compileGLSLprogram(NULL, glsl_draw_fragshader_src);
    //GLint shDrawTex = compileGLSLprogram(glsl_drawtex_vertshader_src, glsl_drawtex_fragshader_src);


    //while (!glfwWindowShouldClose(window)) {
    //    display(input.image_width, input.image_height, map, next_state, input.num_threads, shDrawTex, tex_cudaResult, cuda_dest_resource, cuda_tex_result_resource);
    //    glfwSwapBuffers(window);
    //    glfwPollEvents();
    //    glfwSetKeyCallback(window, key_callback);
    //    Sleep(delay);
    //}
    //glfwTerminate();
    //cudaFree(map);
    //cudaFree(next_state);
    //cudaFree(cuda_dest_resource);



    //bool* next_state = new bool[input.total_size];


    ////return 0;
    window_handler.start();
    window_handler.preprocess_window();
    while (!glfwWindowShouldClose(window_handler.get_window())) {
        display(window_handler);
        glfwSetKeyCallback(window_handler.get_window(), key_callback);
        window_handler.handle_window();
        Sleep(delay);
        //cudaMemcpy(next_state, window_handler.map, sizeof(bool) * input.total_size, cudaMemcpyDeviceToHost);
        //for (int i = 0; i < input.total_size; ++i)
        //    if (next_state[i]) std::cout << i << std::endl;
    }
    window_handler.clean_window();

}


