#include "GUI.h"


// function create texture, set params and register renderbuffer
__host__ void createTextureDst(GLuint* tex_cudaResult, unsigned int size_x, unsigned int size_y, cudaGraphicsResource*& cuda_tex_result_resource) {
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


// function creates shaders (fragment and vertex) using GLEW macros
GLuint compileGLSLprogram(const char* vertex_shader_src, const char* fragment_shader_src) {
    GLuint v, f, p = 0;

    p = glCreateProgram();

    if (vertex_shader_src)
    {
        v = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(v, 1, &vertex_shader_src, NULL);
        glCompileShader(v);

        glAttachShader(p, v);
    }

    if (fragment_shader_src)
    {
        f = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(f, 1, &fragment_shader_src, NULL);
        glCompileShader(f);

        glAttachShader(p, f);
    }

    glLinkProgram(p);

    return p;
}

// display image using OpenGL functions 
void displayImage(GLuint texture, GLint shDrawTex) {
    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, IMAGE_WIDTH_PIXELS, IMAGE_HEIGHT_PIXELS);

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