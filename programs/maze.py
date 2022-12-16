#!/usr/bin/env python3

## @file phong.py
#  Applies the Phong method.
# 
# @author Ricardo Dutra da Silva


import sys
import ctypes
import time
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut
sys.path.append('../lib/')
import utils as ut
from ctypes import c_void_p


## Window width.
win_width  = 1000
## Window height.
win_height = 800

## Program variable.
program = None
## Vertex array object.
VAO = None
## Vertex buffer object.
VBO = None

## Vertex shader.
vertex_code = """
#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform float offsety;
uniform float offsetz;

out vec3 vNormal;
out vec3 fragPosition;

void main()
{
    vec3 offsets;
    offsets.x = 0.0f;
    offsets.y = offsety * 1.0f;
    offsets.z = offsetz * 1.0f;
            
    gl_Position = projection * view * model * vec4(position + offsets, 1.0);
    vNormal = mat3(transpose(inverse(model)))*normal;
    fragPosition = vec3(model * vec4(position, 1.0));
}
"""

## Fragment shader.
fragment_code = """
#version 330 core
in vec3 vNormal;
in vec3 fragPosition;

out vec4 fragColor;

uniform vec3 objectColor;
uniform vec3 lightColor;
uniform vec3 lightPosition;
uniform vec3 cameraPosition;

void main()
{
    float ka = 0.5;
    vec3 ambient = ka * lightColor;

    float kd = 0.8;
    vec3 n = normalize(vNormal);
    vec3 l = normalize(lightPosition - fragPosition);
    
    float diff = max(dot(n,l), 0.0);
    vec3 diffuse = kd * diff * lightColor;

    float ks = 1.0;
    vec3 v = normalize(cameraPosition - fragPosition);
    vec3 r = reflect(-l, n);

    float spec = pow(max(dot(v, r), 0.0), 3.0);
    vec3 specular = ks * spec * lightColor;

    vec3 light = (ambient + diffuse + specular) * objectColor;
    fragColor = vec4(light, 1.0);
}
"""

# Open path to construct maze.
def maze_binary_tree(grid:np.ndarray, size:int) -> np.ndarray:
    output_grid = np.empty([size*3, size*3],dtype=str)
    output_grid[:] = '#'
    
    i = 0
    j = 0
    while i < size:
        w = i*3 + 1
        while j < size:
            k = j*3 + 1
            toss = grid[i,j]
            output_grid[w,k] = ' '
            if toss == 0 and k+2 < size*3:
                output_grid[w,k+1] = ' '
                output_grid[w,k+2] = ' '
            if toss == 1 and w-2 >=0:
                output_grid[w-1,k] = ' '
                output_grid[w-2,k] = ' '
            
            j = j + 1
            
        i = i + 1
        j = 0
        
    return output_grid

# Avoid digging outside the maze external borders
def preprocess_grid(grid:np.ndarray, size:int) -> np.ndarray:
    first_row = grid[0]
    first_row[first_row == 1] = 0
    grid[0] = first_row
    for i in range(1,size):
        grid[i,size-1] = 1
    return grid

## Drawing function.
#
# Draws primitive.
def display():

    gl.glClearColor(0.2, 0.3, 0.3, 1.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    gl.glUseProgram(program)
    gl.glBindVertexArray(VAO)

    S = ut.matScale(0.1, 0.1, 0.1)
    Rx = ut.matRotateX(np.radians(0.0))
    Ry = ut.matRotateY(np.radians(-80.0))
    model = np.matmul(Rx,S)
    model = np.matmul(Ry,model)
    loc = gl.glGetUniformLocation(program, "model");
    gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, model.transpose())
    
    
    projection = ut.matPerspective(np.radians(60.0), win_width/win_height, 0.1, 100.0)
    loc = gl.glGetUniformLocation(program, "projection");
    gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, projection.transpose())
   
    # Light color.
    loc = gl.glGetUniformLocation(program, "lightColor")
    gl.glUniform3f(loc, 1.0, 1.0, 1.0)
    # Light position.
    loc = gl.glGetUniformLocation(program, "lightPosition")
    gl.glUniform3f(loc, 1.0, 0.0, 2.0)
    # Camera position.
    loc = gl.glGetUniformLocation(program, "cameraPosition")
    gl.glUniform3f(loc, 0.0, 0.0, 0.0)

    view = ut.matTranslate(1.0, -0.3, -5.0)
    loc = gl.glGetUniformLocation(program, "view")
    gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, view.transpose())
    
    # Call functions to build and processed grid of maze.
    n=1
    p=0.5
    size=8
    grid = np.random.binomial(n,p, size=(size,size))
    processed_grid = preprocess_grid(grid, size)
    output = maze_binary_tree(processed_grid, size)
    offsety = -20
    offsetz = -20
    
    # Each cube will receive the color relationed with being or not part of
    # the free way of the maze.
    for elm in output:
        for elm2 in elm:
            if(elm2 == '#'):
                loc = gl.glGetUniformLocation(program, "objectColor")
                gl.glUniform3f(loc, 1.0, 1.0, 1.0)
            else:
                loc = gl.glGetUniformLocation(program, "objectColor")
                gl.glUniform3f(loc, 1.5, 1.2, 0.1)
            
            offsety_loc = gl.glGetUniformLocation(program, "offsety")
            gl.glUniform1f(offsety_loc, offsety)
            offsetz_loc = gl.glGetUniformLocation(program, "offsetz")
            gl.glUniform1f(offsetz_loc, offsetz)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 12*3)
            
            if offsetz < 26:
                offsetz += 2
            elif offsetz == 26:
                offsety += 2
                offsetz = -20
    glut.glutSwapBuffers()


## Reshape function.
# 
# Called when window is resized.
#
# @param width New window width.
# @param height New window height.
def reshape(width,height):

    win_width = width
    win_height = height
    gl.glViewport(0, 0, width, height)
    glut.glutPostRedisplay()


## Keyboard function.
#
# Called to treat pressed keys.
#
# @param key Pressed key.
# @param x Mouse x coordinate when key pressed.
# @param y Mouse y coordinate when key pressed.
def keyboard(key, x, y):

    global type_primitive
    global mode
    global vertex_code
    global teste
    global program

    if key == b'\x1b'or key == b'q':
        glut.glutLeaveMainLoop()
        
    program = ut.createShaderProgram(vertex_code, fragment_code)
    glut.glutPostRedisplay()


## Init vertex data.
#
# Defines the coordinates for vertices, creates the arrays for OpenGL.
def initData():

    # Uses vertex arrays.
    global VAO
    global VBO

    # Set triangle vertices.
    vertices = np.array([
        # coordinate       # normal
        -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
         0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
         0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
        -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
         0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
        -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
         0.5, -0.5,  0.5,  1.0,  0.0,  0.0, 
         0.5, -0.5, -0.5,  1.0,  0.0,  0.0,
         0.5,  0.5, -0.5,  1.0,  0.0,  0.0,
         0.5, -0.5,  0.5,  1.0,  0.0,  0.0,
         0.5,  0.5, -0.5,  1.0,  0.0,  0.0,
         0.5,  0.5,  0.5,  1.0,  0.0,  0.0,
         0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
        -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
        -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
         0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
        -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
         0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
        -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,
        -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,
        -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,
        -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,
        -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,
        -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,
        -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
         0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
         0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
        -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
         0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
        -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
        -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
        -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
         0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
        -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
         0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
         0.5, -0.5,  0.5,  0.0, -1.0,  0.0
    ], dtype='float32')

    # Vertex array.
    VAO = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(VAO)

    # Vertex buffer
    VBO = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, VBO)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
    
    # Set attributes.
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 6*vertices.itemsize, None)
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 6*vertices.itemsize, c_void_p(3*vertices.itemsize))
    gl.glEnableVertexAttribArray(1)
    
    # Unbind Vertex Array Object.
    gl.glBindVertexArray(0)

    gl.glEnable(gl.GL_DEPTH_TEST);

## Create program (shaders).
#
# Compile shaders and create programs.
def initShaders():

    global program

    program = ut.createShaderProgram(vertex_code, fragment_code)

## Main function.
#
# Init GLUT and the window settings. Also, defines the callback functions used in the program.
def main():

    glut.glutInit()
    glut.glutInitContextVersion(3, 3);
    glut.glutInitContextProfile(glut.GLUT_CORE_PROFILE);
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH)
    glut.glutInitWindowSize(win_width,win_height)
    glut.glutCreateWindow('Phong')

    # Init vertex data for the triangle.
    initData()
    
    # Create shaders.
    initShaders()

    glut.glutReshapeFunc(reshape)
    glut.glutDisplayFunc(display)
    glut.glutKeyboardFunc(keyboard)

    glut.glutMainLoop()

if __name__ == '__main__':
    main()
