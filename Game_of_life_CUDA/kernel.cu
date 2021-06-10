
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GLUT_DISABLE_ATEXIT_HACK
//#include <stdio.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <Windows.h>
#include <stdio.h>
//#include <gl/glut.h>
#include <helper_gl.h>
#include <helper_cuda.h>
#include <cuda_gl_interop.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <GL/freeglut.h>
#include <GLFW/glfw3.h>

#define GLFW_KEY_U 85
#define GLFW_KEY_D 68
constexpr u_int max_num_blocks = 32768;

using u_int = unsigned int;
using u_short = unsigned short;


//int image_width = 512;
//int image_height = 512;

GLint shDrawTex = -1;
GLint shDraw = -1;
struct cudaGraphicsResource* cuda_tex_result_resource;
struct cudaGraphicsResource* cuda_dest_resource;
GLuint tex_cudaResult;

//bool* map;
//bool* next_state;
//u_int num_threads;

__global__ void next_step(bool* map, bool* next_state, u_int width, u_int length, int32_t* image ) {
    u_int total_size = width * length;

    for (u_int idx = (u_int)(blockIdx.x * blockDim.x + threadIdx.x); idx < total_size; idx += (u_int)(blockDim.x * gridDim.x)) {
        // now we need to organize 2d in 1d array
        // for each element in map array we want to calculate number of alive neighbours
        // so we need to figure out how to indexing through given array
        // second problem is that we need to compute this but with respect of "curved borders" of the map

        // x positions of actual idx and right and left neighbour
        u_int x = idx % width;
        u_int x_left = (x + width - 1) % width;
        u_int x_right = (x + 1) % width;

        // y coords analogically as x
        u_int y = idx - x;
        u_int y_top = (y + total_size - width) % total_size;
        u_int y_bottom = (y + width) % total_size;

        u_short num_neigbhbour_alive = 0;

        // top neighbours
        num_neigbhbour_alive += (u_short)map[x_left + y_top] + (u_short)map[x + y_top] + (u_short)map[x_right + y_top];
        // left & right neighbours
        num_neigbhbour_alive += (u_short)map[x_left + y] + (u_short)map[x_right + y];
        // bottom neighbours
        num_neigbhbour_alive += (u_short)map[x_left + y_bottom] + (u_short)map[x + y_bottom] + (u_short)map[x_right + y_bottom];

        // compute state and set color
        next_state[x + y] = num_neigbhbour_alive == 3 || (num_neigbhbour_alive == 2 && map[x + y]);
        image[x + y] = next_state[x + y] ? -1 : 0;
    }
}


//__host__ void print_world(bool* map, u_int width, u_int length) {
//    for (u_int i = 0; i < width; ++i) {
//        for (u_int j = 0; j < length; ++j) {
//            char sign;
//            map[i * width + j] ? sign = '*' : sign = ' ';
//            std::cout << sign;
//            //printf("\r%c", sign);
//        }
//        std::cout<<std::endl;
//        //printf("\r\n");
//    }
//}

//__host__ void calculate_map(bool*& map, bool*& next_state, u_int width, u_int length, u_short num_threads) {
//    assert(width * length % num_threads == 0);
//    u_int requested_blocks = ((width * length) / num_threads);
//    u_int num_blocks = (u_int)min(max_num_blocks, requested_blocks);
//    //next_step << <num_blocks, num_threads >> > (map, next_state, width, length);
//    std::swap(map, next_state);
//}
// 
__host__ void calculate_map(bool*& map, bool*& next_state, u_int image_width, u_int image_height, u_short num_threads, int32_t*&out_data) {
    assert(image_width * image_height % num_threads == 0);
    u_int requested_blocks = ((image_width * image_height) / num_threads);
    u_int num_blocks = (u_int)min(max_num_blocks, requested_blocks);
    next_step <<< num_blocks, num_threads >>> (map, next_state, image_width, image_height, out_data);
    std::swap(map, next_state);
}

__host__ void createTextureDst(GLuint* tex_cudaResult, unsigned int size_x, unsigned int size_y)
{
    // this section creates texture
    glGenTextures(1, tex_cudaResult);
    glBindTexture(GL_TEXTURE_2D, *tex_cudaResult);

    // params setting
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, size_x, size_y, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
    
    // register given texture using CUDA
    cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, *tex_cudaResult,
        GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
}

GLuint compileGLSLprogram(const char* vertex_shader_src, const char* fragment_shader_src)
{
    GLuint v, f, p = 0;

    p = glCreateProgram();

    if (vertex_shader_src)
    {
        v = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(v, 1, &vertex_shader_src, NULL);
        glCompileShader(v);

        GLint compiled = 0;
        glGetShaderiv(v, GL_COMPILE_STATUS, &compiled);

        if (!compiled)
        {
            char temp[256] = "";
            glGetShaderInfoLog(v, 256, NULL, temp);
            printf("Vtx Compile failed:\n%s\n", temp);
            glDeleteShader(v);
            return 0;
        }
        else
        {
            glAttachShader(p, v);
        }
    }

    if (fragment_shader_src)
    {
        f = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(f, 1, &fragment_shader_src, NULL);
        glCompileShader(f);

        GLint compiled = 0;
        glGetShaderiv(f, GL_COMPILE_STATUS, &compiled);

        if (!compiled)
        {
            char temp[256] = "";
            glGetShaderInfoLog(f, 256, NULL, temp);
            printf("frag Compile failed:\n%s\n", temp);
            glDeleteShader(f);
            return 0;
        }
        else
        {
            glAttachShader(p, f);
        }
    }

    glLinkProgram(p);

    int infologLength = 0;
    int charsWritten = 0;

    glGetProgramiv(p, GL_INFO_LOG_LENGTH, (GLint*)&infologLength);

    if (infologLength > 0)
    {
        char* infoLog = (char*)malloc(infologLength);
        glGetProgramInfoLog(p, infologLength, (GLsizei*)&charsWritten, infoLog);
        printf("Shader compilation error: %s\n", infoLog);
        free(infoLog);
    }

    return p;
}

void initCUDABuffers(u_int image_width, u_int image_height)
{
    // set up vertex data parameter
    int num_texels = image_width * image_height;
    int num_values = num_texels * 4;
    size_t size_tex_data = sizeof(GLubyte) * num_values;
    cudaMalloc((void**)&cuda_dest_resource, size_tex_data);
    //checkCudaErrors(cudaHostAlloc((void**)&cuda_dest_resource, size_tex_data, ));
}


const GLenum fbo_targets[] =
{
    GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT,
    GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT
};

static const char* glsl_drawtex_vertshader_src =
"void main(void)\n"
"{\n"
"	gl_Position = gl_Vertex;\n"
"	gl_TexCoord[0].xy = gl_MultiTexCoord0.xy;\n"
"}\n";

static const char* glsl_drawtex_fragshader_src =
"#version 130\n"
"uniform usampler2D texImage;\n"
"void main()\n"
"{\n"
"   vec4 c = texture(texImage, gl_TexCoord[0].xy);\n"
"	gl_FragColor = c / 255.0;\n"
"}\n";

static const char* glsl_draw_fragshader_src =
//WARNING: seems like the gl_FragColor doesn't want to output >1 colors...
//you need version 1.3 so you can define a uvec4 output...
//but MacOSX complains about not supporting 1.3 !!
// for now, the mode where we use RGBA8UI may not work properly for Apple : only RGBA16F works (default)
"#version 130\n"
"out uvec4 FragColor;\n"
"void main()\n"
"{"
"  FragColor = uvec4(gl_Color.xyz * 255.0, 255.0);\n"
"}\n";

// copy image and process using CUDA
void generateCUDAImage(u_int image_width, u_int image_height, bool*& map, bool*& next_state, u_int num_threads)
{
    // run the Cuda kernel
    int32_t* out_data;

    out_data = reinterpret_cast<int32_t*>(cuda_dest_resource);
    // calculate grid size
    dim3 block(16, 16, 1);
    //dim3 block(16, 16, 1);
    dim3 grid(image_width / block.x, image_height / block.y, 1);
    // execute CUDA kernel
    //launch_cudaProcess(grid, block, 0, out_data, image_width);
 

    //assert(image_width * image_height % num_threads == 0);
    //u_int requested_blocks = ((image_width * image_height) / num_threads);
    //u_int num_blocks = (u_int)min(max_num_blocks, requested_blocks);
    //next_step << <num_blocks, num_threads >> > (map, next_state, image_width, image_height, out_data);
    //std::swap(map, next_state);
    calculate_map(map, next_state, image_width, image_height, num_threads, out_data);



    cudaArray* texture_ptr;
    cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0);
    cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0);

    int num_texels = image_width * image_height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource, size_tex_data, cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0);
}

// display image to the screen as textured quad
void displayImage(GLuint texture)
{
    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    //glDisable(GL_LIGHTING);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, 512, 512);

    glUseProgram(shDrawTex);
    GLint id = glGetUniformLocation(shDrawTex, "texImage");
    glUniform1i(id, 0);


    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(-1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 0.0);
    glVertex3f(1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 1.0);
    glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(0.0, 1.0);
    glVertex3f(-1.0, 1.0, 0.5);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);

    glUseProgram(0);
}


void
display(u_int image_width, u_int image_height, bool*& map, bool*& next_state, u_int num_threads)
{
    generateCUDAImage(image_width, image_height, map, next_state, num_threads);
    displayImage(tex_cudaResult);
    cudaDeviceSynchronize();
    //glutSwapBuffers();
}

void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

u_int delay = 250;
void speed_up() {
    delay = (int)delay * 0.9;
}

void slow_down() {
    delay = (int)delay * 1 / (0.9);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_U && action == GLFW_PRESS)
        speed_up();
    if (key == GLFW_KEY_D && action == GLFW_PRESS)
        slow_down();
}



int main(int argc, char** argv) {
    u_int image_width, image_height, total_size, num_threads;
    bool *map, *next_state;
    bool *host_map, *host_next_state;
    bool* result;
    std::cout << "Welcome to the game of life!" << std::endl;
    std::cout << "Please choose f if you want to load data from file or press m to input data manually." << std::endl;
    input:
    char mode = getchar();
    if (mode != 'f' && mode != 'm') {
        std::cout << "You choose wrong mode: " << mode << ". Please use f for file or m for manually." << std::endl;
        goto input;
    }
    else if (mode == 'f') {
        std::cout << "Now put the name of the file with size and start positions" << std::endl;
        std::string file_name;
        std::cin >> file_name;
        
        std::ifstream infile(file_name.c_str());
        //infile.open(file_name);
        if (!infile)
        {
            std::cout << "No dupa nie otwiera się" << std::endl;
        }
        if (!(infile >> num_threads)) {
            std::cout << num_threads << std::endl;
            std::cout << "Wrong file format, file must include number of threads" << std::endl;
            exit(1);
        }
        if (!(infile >> image_width >> image_height)) {
            std::cout << "Wrong file format, file must include width and height" << std::endl;
            exit(1);
        }
        total_size = image_width * image_height;
        //cudaMalloc((void**)&host_map, sizeof(bool)*total_size);
        //cudaMalloc((void**)&host_next_state, sizeof(bool)*total_size);
        host_map = new bool[total_size];
        host_next_state = new bool[total_size];
        for (u_int i = 0; i < total_size; ++i) 
            host_map[i] = host_next_state[i] = false;
        u_int num_start_alive;
        if (!(infile >> num_start_alive)) {
            std::cout << "Wrong file format, file must include number of alive cells at the beginning" << std::endl;
            exit(1);
        }
        u_int x_pos, y_pos;
        u_int num_loaded = 0;
        while (infile >> x_pos >> y_pos){
            if (x_pos >= image_width)
                x_pos = x_pos % image_width;
            if (y_pos >= image_height)
                y_pos = y_pos % image_height;
            num_loaded += 2;
            host_map[x_pos * image_width + y_pos] = true;
        }
        if (num_loaded < 2 * num_start_alive) {
            std::cout << "File does not contains enough coords" << std::endl;
            /*cudaFree(map);
            cudaFree(next_state);*/
            exit(1);
        }
        infile.close();
    }
    else {
        std::cout << "Please input number of threads" << std::endl;
        std::cin >> num_threads;
        std::cout << "Now, please input width and length"<< std::endl;
        std::cin >> image_width >> image_height;
        total_size = image_width * image_height;
        //cudaMalloc((void**)&host_map, sizeof(bool) * total_size);
        //cudaMalloc((void**)&host_next_state, sizeof(bool) * total_size);
        host_map = new bool[total_size];
        host_next_state = new bool[total_size];
        //cudaMemset(host_map, false, total_size);
        //cudaMemset(host_next_state, false, total_size);
        for (u_int i = 0; i < total_size; ++i)
            host_map[i] = host_next_state[i] = false;
        u_int counter = 0;
        u_int x_pos, y_pos;
        u_int num_start_alive;
        std::cout << "Input number of alive cells at the beginning" << std::endl;
        std::cin >> num_start_alive;
        std::cout << "Input starting data" << std::endl;
        while (counter != num_start_alive) {
            std::cin >> x_pos >> y_pos;
            if (x_pos >= image_width)
                x_pos = x_pos % image_width;
            if (y_pos >= image_height)
                y_pos = y_pos % image_height;
            counter++;
            host_map[x_pos * image_width + y_pos] = true;
        }
    }

    //image_width = 32;
    //image_height = 32;
    //u_int total_size = image_width * image_height;
    //num_threads = 32;
    //host_map = new bool[total_size];
    //host_next_state = new bool[total_size];
    //for (u_int i = 0; i < total_size; ++i)
    //    host_map[i] = host_next_state[i] = false;
    //host_map[7 * image_width + 7] = true;
    //host_map[7 * image_width + 8] = true;
    //host_map[7 * image_width + 9] = true;
    //host_map[6 * image_width + 6] = true;
    //host_map[6 * image_width + 7] = true;
    //host_map[6 * image_width + 8] = true;
    cudaMalloc((void**)&map, sizeof(bool) * total_size);
    cudaMalloc((void**)&next_state, sizeof(bool) * total_size);
    cudaMalloc((void**)&cuda_dest_resource, sizeof(int32_t) * total_size);
    //cudaMalloc((void**)&result, sizeof(bool) * total_size);
    cudaMemset(map, false, total_size);
    cudaMemset(next_state, false, total_size);
    //host_map = new bool[total_size];
    std::cout << "Do you want to start [Y/n]?" << std::endl;
    char decision;
    std::cin >> decision;
    if (decision != 'y' && decision != 'Y') {
        std::cout << "Ending" << std::endl;
        return 0;
    }


    //glutInit(&argc, argv);
    //glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    //glutInitWindowSize(512, 512);
    //int iGLUTWindowHandle = glutCreateWindow("CUDA OpenGL post-processing");
    glfwSetErrorCallback(error_callback);
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(512, 512, "Okienko", NULL, NULL);
    glfwMakeContextCurrent(window);
    glClearColor(0.5, 0.5, 0.5, 1.0);
    glDisable(GL_DEPTH_TEST);

    //// viewport
    //glViewport(0, 0, image_width, image_height);

    // projection
    //glMatrixMode(GL_PROJECTION);
    //glLoadIdentity();
    //gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1f, 10.0f);
    //glOrtho(-1, 1, -1, 1, 0.001, 1);
    cudaMemcpy(map, host_map, sizeof(bool) * total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(next_state, host_next_state, sizeof(bool) * total_size, cudaMemcpyHostToDevice);
    createTextureDst(&tex_cudaResult, image_width, image_height);

    glewInit();
    //glEnable(GL_LIGHT0);
    //float red[] = { 1.0f, 0.1f, 0.1f, 1.0f };
    //float white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    //glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
    //glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
    //glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0f);
    shDraw = compileGLSLprogram(NULL, glsl_draw_fragshader_src);
    shDrawTex = compileGLSLprogram(glsl_drawtex_vertshader_src, glsl_drawtex_fragshader_src);
    //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    while (!glfwWindowShouldClose(window)) {
        display(image_width, image_height, map, next_state, num_threads);
        glfwSwapBuffers(window);
        glfwPollEvents();
        glfwSetKeyCallback(window, key_callback);
        Sleep(delay);
    }
    glfwTerminate();
    //glutDisplayFunc(display);
    //glutMainLoop();

    //while (true) {
    //    cudaMemcpy(map, host_map, sizeof(bool) * total_size, cudaMemcpyHostToDevice);
    //    cudaMemcpy(next_state, host_next_state, sizeof(bool) * total_size, cudaMemcpyHostToDevice);

    //    calculate_map(map, next_state, width, length, num_threads);

    //    cudaMemcpy(host_map, map, sizeof(bool) * total_size, cudaMemcpyDeviceToHost);


    //    print_world(host_map, width, length);
    //    //glutInit(&argc, argv);
    //    //glutInitWindowSize(640, 480);
    //    //glutInitWindowPosition(10, 10);
    //    //glutCreateWindow("User_Name");
    //    //myinit();
    //    //glutDisplayFunc(display);
    //    //glutMainLoop();
    //    //return 0;
    //    Sleep(delay);
    //    //printf("\r");
    //    //fflush(stdout);
    //    //flush_screen();
    //}
    return 0;
}