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

#define IMAGE_WIDTH_PIXELS 800
#define IMAGE_HEIGHT_PIXELS 800

__host__ void createTextureDst(GLuint* tex_cudaResult, unsigned int size_x, unsigned int size_y, cudaGraphicsResource*& cuda_tex_result_resource);

GLuint compileGLSLprogram(const char* vertex_shader_src, const char* fragment_shader_src);

void displayImage(GLuint texture, GLint shDrawTex);

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

