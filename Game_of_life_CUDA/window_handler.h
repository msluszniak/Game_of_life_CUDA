#pragma once
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

class Window_handler {
	struct inputer_data input;
	GLFWwindow* window;
	GLuint tex_cudaResult;
	GLint shDrawTex;
	int delay = 250;

public:
	struct cudaGraphicsResource* cuda_dest_resource;
	struct cudaGraphicsResource* cuda_tex_result_resource = NULL;
	bool* map, * next_state;

	explicit Window_handler(inputer_data user_input) : input{ user_input } {
		cudaMalloc((void**)&map, sizeof(bool) * input.total_size);
		cudaMalloc((void**)&next_state, sizeof(bool) * input.total_size);
		cudaMalloc((void**)&cuda_dest_resource, sizeof(int32_t) * input.total_size);
		cudaMemset(map, false, input.total_size);
		cudaMemset(next_state, false, input.total_size);
	}

public:
	void start() {
		std::cout << "Do you want to start [Y/n]?" << std::endl;
		char decision;
		std::cin >> decision;
		if (decision != 'y' && decision != 'Y') {
			std::cout << "Ending" << std::endl;
			return;
		}
	}

	void error_callback(int error, const char* description) {
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


	void preprocess_window() {
		//glfwSetErrorCallback(error_callback);
		glfwInit();
		window = glfwCreateWindow(IMAGE_WIDTH_PIXELS, IMAGE_HEIGHT_PIXELS, "Okienko", NULL, NULL);
		glfwMakeContextCurrent(window);
		glClearColor(0.5, 0.5, 0.5, 1.0);
		glDisable(GL_DEPTH_TEST);

		cudaMemcpy(map, input.host_map, sizeof(bool) * input.total_size, cudaMemcpyHostToDevice);
		cudaMemcpy(next_state, input.host_next_state, sizeof(bool) * input.total_size, cudaMemcpyHostToDevice);
		createTextureDst(&tex_cudaResult, input.image_width, input.image_height, cuda_tex_result_resource);

		glewInit();
		shDrawTex = compileGLSLprogram(glsl_drawtex_vertshader_src, glsl_drawtex_fragshader_src);
	}

	void handle_window() {
		glfwSwapBuffers(window);
		glfwPollEvents();
		//Sleep(delay);
	}

	void clean_window() {
		glfwTerminate();
		cudaFree(map);
		cudaFree(next_state);
		cudaFree(cuda_dest_resource);
	}

	GLFWwindow* get_window() {
		return window;
	}

	GLuint get_tex_cudaResult() {
		return tex_cudaResult;
	}

	GLint get_shDrawTex() {
		return shDrawTex;
	}

	cudaGraphicsResource* get_cuda_dest_resource() {
		return cuda_dest_resource;
	}

	inputer_data get_input() {
		return input;
	}
};