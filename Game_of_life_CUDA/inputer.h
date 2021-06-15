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


struct inputer_data {
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


//
//void inputer(u_int& num_threads, u_int& image_width, u_int& image_height, u_int& total_size, bool*& host_map, bool*& host_next_state) {
//    std::cout << "Welcome to the game of life!" << std::endl;
//    std::cout << "Please choose f if you want to load data from file or press m to input data manually." << std::endl;
//input:
//    char mode = getchar();
//    if (mode != 'f' && mode != 'm') {
//        std::cout << "You choose wrong mode: " << mode << ". Please use f for file or m for manually." << std::endl;
//        goto input;
//    }
//    else if (mode == 'f') {
//        std::cout << "Now put the name of the file with size and start positions" << std::endl;
//        std::string file_name;
//        std::cin >> file_name;
//
//        std::ifstream infile(file_name.c_str());
//        if (!infile){
//            std::cout << "Cannot open file correctly" << std::endl;
//        }
//        if (!(infile >> num_threads)) {
//            std::cout << num_threads << std::endl;
//            std::cout << "Wrong file format, file must include number of threads" << std::endl;
//            exit(1);
//        }
//        if (!(infile >> image_width >> image_height)) {
//            std::cout << "Wrong file format, file must include width and height" << std::endl;
//            exit(1);
//        }
//        total_size = image_width * image_height;
//        host_map = new bool[total_size];
//        host_next_state = new bool[total_size];
//        for (u_int i = 0; i < total_size; ++i)
//            host_map[i] = host_next_state[i] = false;
//        u_int num_start_alive;
//        if (!(infile >> num_start_alive)) {
//            std::cout << "Wrong file format, file must include number of alive cells at the beginning" << std::endl;
//            exit(1);
//        }
//        u_int x_pos, y_pos;
//        u_int num_loaded = 0;
//        while (infile >> x_pos >> y_pos) {
//            if (x_pos >= image_width)
//                x_pos = x_pos % image_width;
//            if (y_pos >= image_height)
//                y_pos = y_pos % image_height;
//            num_loaded += 2;
//            host_map[x_pos * image_width + y_pos] = true;
//        }
//        if (num_loaded < 2 * num_start_alive) {
//            std::cout << "File does not contains enough coords" << std::endl;
//            delete[] host_map;
//            delete[] host_next_state;
//            exit(1);
//        }
//        infile.close();
//    }
//    else {
//        std::cout << "Please input number of threads" << std::endl;
//        std::cin >> num_threads;
//        std::cout << "Now, please input width and length" << std::endl;
//        std::cin >> image_width >> image_height;
//        total_size = image_width * image_height;
//        host_map = new bool[total_size];
//        host_next_state = new bool[total_size];
//        for (u_int i = 0; i < total_size; ++i)
//            host_map[i] = host_next_state[i] = false;
//        u_int counter = 0;
//        u_int x_pos, y_pos;
//        u_int num_start_alive;
//        std::cout << "Input number of alive cells at the beginning" << std::endl;
//        std::cin >> num_start_alive;
//        std::cout << "Input starting data" << std::endl;
//        while (counter != num_start_alive) {
//            std::cin >> x_pos >> y_pos;
//            if (x_pos >= image_width)
//                x_pos = x_pos % image_width;
//            if (y_pos >= image_height)
//                y_pos = y_pos % image_height;
//            counter++;
//            host_map[x_pos * image_width + y_pos] = true;
//        }
//    }
//}