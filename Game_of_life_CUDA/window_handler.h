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


//class for app window
class Window_handler {
	struct inputer_data input;
	GLFWwindow* window;
	GLuint tex_cudaResult;
	GLint shDrawTex;

public:
	struct cudaGraphicsResource* cuda_dest_resource;
	struct cudaGraphicsResource* cuda_tex_result_resource = NULL;
	bool* map, * next_state;

	explicit Window_handler(inputer_data user_input);
	void start();
	void preprocess_window();
	void handle_window();
	void clean_window();

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