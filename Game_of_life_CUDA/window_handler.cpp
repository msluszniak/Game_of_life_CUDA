#include "window_handler.h"

//constructor allocates memory for map arrays and GUI array
Window_handler::Window_handler(inputer_data user_input) : input{ user_input } {
	cudaMalloc((void**)&map, sizeof(bool) * input.total_size);
	cudaMalloc((void**)&next_state, sizeof(bool) * input.total_size);
	cudaMalloc((void**)&cuda_dest_resource, sizeof(int32_t) * input.total_size);
	cudaMemset(map, false, input.total_size);
	cudaMemset(next_state, false, input.total_size);
}

// confirm if user wants to start the simulation
void Window_handler::start() {
	std::cout << "Do you want to start [Y/n]?" << std::endl;
	char decision;
	std::cin >> decision;
	if (decision != 'y' && decision != 'Y') {
		std::cout << "Ending" << std::endl;
		return;
	}
}


void Window_handler::preprocess_window() {
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

// swap images (buffers) and process pending events
void Window_handler::handle_window() {
	glfwSwapBuffers(window);
	glfwPollEvents();
}

// clean all allocated arrays and OpenGL objects
void Window_handler::clean_window() {
	glfwTerminate();
	cudaFree(map);
	cudaFree(next_state);
	cudaFree(cuda_dest_resource);
	delete[] input.host_map;
	delete[] input.host_next_state;
}