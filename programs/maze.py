#!/usr/bin/env python3

## @file maze.py
#  Generates a maze using graph algorithm.
# 
# @author Ricardo Dutra da Silva


import sys
import ctypes
from ctypes import c_void_p
import time
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut
sys.path.append('../lib/')
import utils as ut
import random
import heapq


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

algorithm = 1

camera_posx = 0.0
camera_posy = 0.0

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

# function to generate the maze
def maze_kruskal(n, m):
  def find(p, q):
    if p != cells[p] or q != cells[q]:
      cells[p], cells[q] = find(cells[p], cells[q])
    return cells[p], cells[q]    # find spanning tree

  # make even size to avoid issues in the future
  if n % 2 == 1:
    n = n + 1
  if m % 2 == 1:
    m = m + 1

  # maze checkerboard of wall squares and squares that can be either walls or pathways
  maze = np.tile([[1, 2], [2, 0]], (n // 2 + 1, m // 2 + 1))
  maze = maze[:-1, :-1]
  cells = {(i, j): (i, j) for i, j in np.argwhere(maze == 1)}
  walls = np.argwhere(maze == 2)    # union-find
  np.random.shuffle(walls)

  # kruskal's maze algorithm
  for wi, wj in walls:
    if wi % 2:
      p, q = find((wi - 1, wj), (wi + 1, wj))
    else:
      p, q = find((wi, wj - 1), (wi, wj + 1))
    maze[wi, wj] = p != q
    if p != q:
      cells[p] = q

  # initialise the vertical borders
  vertBordersLeft = np.zeros((n + 3,), dtype=int)
  vertBordersRight = vertBordersLeft.copy()
  vertBordersLeft[random.randint(1, n // 2) * 2 - 1] = 6 # Create entrance where agent starts
  vertBordersRight[random.randint(1, n // 2) * 2 - 1] = 5 # Create exit
  
  # initialise the horizontal borders
  horiBorders = np.zeros((m + 1,), dtype=int)

  # pad maze with walls
  maze = np.concatenate(([horiBorders], maze), axis=0)
  maze = np.concatenate((maze, [horiBorders]), axis=0)
  maze = np.insert(maze, 0, vertBordersLeft, axis=1)
  maze = np.insert(maze, len(maze[0]), vertBordersRight, axis=1)

  return maze

def maze_int_kruskal(maze):
    output_int = [[int(-2) for col in row] for row in maze]
    z = 0
    w = 0
    starti = 0
    endi = 0
    for z in range(0,24):
        for w in range(0,24):
            if maze[z][w] == 0:
                output_int[z][w] = 1
            elif maze[z][w] == 1 or maze[z][w] == 4 or maze[z][w] == 5:
                output_int[z][w] = 0
            if maze[z][w] == 6:
                starti = z
            if maze[z][w] == 5:
                endi = z
    
    distance, prev_point = dijkstra(output_int, (starti, 0))
    dij = find_shortest_path(prev_point, (endi, 25))
    
    for elm in dij:
        output_int[elm[0]][elm[1]] = 2
    
    return output_int

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

def maze_int_binary_tree(maze):
    output_int = [[int(0) for col in row] for row in maze]
    z = 0
    w = 0
    starti = 0
    endi = 0
    for z in range(0,24):
        for w in range(0,24):
            if maze[z][w] == '#':
                output_int[z][w] = 1
                
    distance, prev_point = dijkstra(output_int, (22, 22))
    dij = find_shortest_path(prev_point, (19, 1))
    
    for elm in dij:
        output_int[elm[0]][elm[1]] = 2
    
    return output_int

def dijkstra(graph, start_point):
    # initialize graphs to track if a point is visited,
    # current calculated distance from start to point,
    # and previous point taken to get to current point
    visited = [[False for col in row] for row in graph]
    z = 0
    w = 0
    for z in range(0,24):
        for w in range(0,24):
            if graph[z][w] == 1:
                visited[z][w] = True
    distance = [[float('inf') for col in row] for row in graph]
    distance[start_point[0]][start_point[1]] = 0
    prev_point = [[None for col in row] for row in graph]
    n, m = len(graph), len(graph[0])
    number_of_points, visited_count = n * m, 0
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    min_heap = []

    # min_heap item format:
    # (pt's dist from start on this path, pt's row, pt's col)
    heapq.heappush(min_heap, (distance[start_point[0]][start_point[1]], start_point[0], start_point[1]))

    while visited_count < number_of_points:
        if min_heap != []:
            current_point = heapq.heappop(min_heap)
            distance_from_start, row, col = current_point
        for direction in directions:
            new_row, new_col = row + direction[0], col + direction[1]
            if -1 < new_row < n and -1 < new_col < m and not visited[new_row][new_col]:
                dist_to_new_point = distance_from_start + graph[new_row][new_col]
                if dist_to_new_point < distance[new_row][new_col]:
                    distance[new_row][new_col] = dist_to_new_point
                    prev_point[new_row][new_col] = (row, col)
                    heapq.heappush(min_heap, (dist_to_new_point, new_row, new_col))
        visited[row][col] = True
        visited_count += 1

    return distance, prev_point

def find_shortest_path(prev_point_graph, end_point):
    shortest_path = []
    current_point = end_point
    i = 0
    while current_point is not None:
        i += 1
        shortest_path.append(current_point)
        current_point = prev_point_graph[current_point[0]][current_point[1]]
    shortest_path.reverse()
    
    print("Solved Maze in", i, "steps.")
    
    return shortest_path

## Drawing function.
#
# Draws primitive.
def display():
    
    global algorithm
    global camera_posx
    global camera_posy
    
    gl.glClearColor(0.2, 0.3, 0.3, 1.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    gl.glUseProgram(program)
    gl.glBindVertexArray(VAO)

    S = ut.matScale(0.1, 0.1, 0.1)
    Rx = ut.matRotateX(np.radians(0.0))
    Ry = ut.matRotateY(np.radians(-80.0+camera_posx))
    model = np.matmul(Rx,S)
    model = np.matmul(Ry,model)
    loc = gl.glGetUniformLocation(program, "model");
    gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, model.transpose())
    
    
    projection = ut.matPerspective(np.radians(camera_posy+60.0), win_width/win_height, 0.1, 100.0)
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
    gl.glUniform3f(loc, 0.0, camera_posx, camera_posy)

    view = ut.matTranslate(1.0, -0.3, -5.0)
    loc = gl.glGetUniformLocation(program, "view")
    gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, view.transpose())
    
    if algorithm == 1:
        n=1
        p=0.5
        size=8
        grid = np.random.binomial(n,p, size=(size,size))
        processed_grid = preprocess_grid(grid, size)
        maze = maze_binary_tree(processed_grid, size)
        output_int = maze_int_binary_tree(maze)
    elif algorithm == 2:
        x = 25
        y = 25
        maze = maze_kruskal(x, y)
        output_int = maze_int_kruskal(maze)
    
    # Each cube will receive the color relationed with being or not part of
    # the free way of the maze.
    offsety = -20
    offsetz = -20
    z = 0
    w = 0
    for z in range(0,24):
        for w in range(0,24):
            if(output_int[z][w] == 1):
                loc = gl.glGetUniformLocation(program, "objectColor")
                gl.glUniform3f(loc, 1.0, 1.0, 1.0)
            elif (output_int[z][w] == 0):
                loc = gl.glGetUniformLocation(program, "objectColor")
                gl.glUniform3f(loc, 1.5, 0.2, 0.2)
            else:
                loc = gl.glGetUniformLocation(program, "objectColor")
                gl.glUniform3f(loc, 0.2, 2.0, 0.8)
            
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

    global algorithm
    global camera_posx
    global camera_posy

    if key == b'\x1b'or key == b'q':
        glut.glutLeaveMainLoop()
    
    if key == b't'or key == b'T':
        algorithm = 1
    
    if key == b'k'or key == b'K':
        algorithm = 2
    
    if key == b'w' or key == b'W':
        camera_posy += 0.5;
    if key == b's' or key == b'S':
        camera_posy -= 0.5;
    if key == b'a' or key == b'A':
        camera_posx -= 0.5;
    if key == b'd' or key == b'D':
        camera_posx += 0.5;
    
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
    glut.glutCreateWindow('Maze Generator Using Graph Algorithm')
    
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
