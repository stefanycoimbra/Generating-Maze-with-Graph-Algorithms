#!/usr/bin/env python3

## @file maze.py
# Generates a maze using graph algorithm.
# 
# Adapted code of Ricardo Dutra da Silva.
# @authors: Stéfany Coura Coimbra - 2019008562
#           Viviane Cardosina Cordeiro - 2019004984
#           Ytalo Ysmaicon Gomes - 2019000223

import sys
import ctypes
from ctypes import c_void_p
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

## Variable to choose the graph algorithm to generate maze: Kruskal or Binary Tree.
algorithm = 1

## Camera Position's variables  
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

## Maze Kruskal function.
#
# Generate the maze using kruskal.
#
# @param size_row Size of rows.
# @param size_column Size of columns.
def maze_kruskal(size_row, size_column):

  ## Union-Find internal function.
  #
  # Union and Find recursively p and q sets.
  #
  # @param p First set.
  # @param q Second set. 
  def union_find(p, q):
    
    if p != cells[p] or q != cells[q]:
      cells[p], cells[q] = union_find(cells[p], cells[q])
    return cells[p], cells[q]    

  # Make even size of matrix to work with that later.
  if size_row % 2 == 1:
    size_row = size_row + 1
  if size_column % 2 == 1:
    size_column = size_column + 1

  # Create and initialize maze with some walls randomly.
  maze = np.tile([[1, 2], [2, 0]], (size_row // 2 + 1, size_column // 2 + 1))
  maze = maze[:-1, :-1]
  cells = {(i, j): (i, j) for i, j in np.argwhere(maze == 1)}
  walls = np.argwhere(maze == 2)    
  np.random.shuffle(walls)

  # Find spanning tree.
  for wall_i, wall_j in walls:
    if wall_i % 2:
      p, q = union_find((wall_i - 1, wall_j), (wall_i + 1, wall_j))
    else:
      p, q = union_find((wall_i, wall_j - 1), (wall_i, wall_j + 1))
    maze[wall_i, wall_j] = p != q
    if p != q:
      cells[p] = q

  # Initialise the vertical borders.
  verticalBordersLeft = np.zeros((size_row + 3,), dtype=int)
  verticalBordersRight = verticalBordersLeft.copy()
  # Create entrance where agent starts.
  verticalBordersLeft[random.randint(1, size_row // 2) * 2 - 1] = 6
  # Create exit. 
  verticalBordersRight[random.randint(1, size_row // 2) * 2 - 1] = 5 
  
  # Initialise the horizontal borders.
  horizontalBorders = np.zeros((size_column + 1,), dtype=int)

  # Concatenate the borders in the maze.
  maze = np.concatenate(([horizontalBorders], maze), axis=0)
  maze = np.concatenate((maze, [horizontalBorders]), axis=0)
  maze = np.insert(maze, 0, verticalBordersLeft, axis=1)
  maze = np.insert(maze, len(maze[0]), verticalBordersRight, axis=1)

  return maze

## Maze Kruskal Int Pattern.
#
# Generate matrix with 1-wall, 0-free path, 2-cell part of the shortest path. 
#
# @param maze matrix with the set values to build the scene.
def maze_int_kruskal(maze):

    output_int = [[int(-2) for col in row] for row in maze]
    z = 0
    w = 0
    starti = 0
    endi = 0
    for z in range(0, 24):
        for w in range(0, 24):
            if maze[z][w] == 0:
                output_int[z][w] = 1
            elif maze[z][w] == 1 or maze[z][w] == 4 or maze[z][w] == 5:
                output_int[z][w] = 0
            if maze[z][w] == 6:
                starti = z
            if maze[z][w] == 5:
                endi = z
    
    distance, prev_point = dijkstra(output_int, (starti, 0))
    dij = find_shortest_path(prev_point, (endi, 24))
    
    for elm in dij:
        output_int[elm[0]][elm[1]] = 2
    
    return output_int

## Maze Binary Three function.    
#
# Open path to construct maze using the algorithm binary tree.
#
# @param grid Grid.
# #param size Length of three.
def maze_binary_tree(grid, size):

    output_grid = np.empty([size*3, size*3], dtype=str)
    output_grid[:] = '#'
    
    i = 0
    j = 0
    while i < size:
        w = i*3 + 1
        while j < size:
            k = j*3 + 1
            direction = grid[i, j]
            output_grid[w, k] = ' '
            # The grid is generated in every interaction of the user to build a maze.
            # Direction taken -> north/east.
            if direction == grid[i, j] == 0 and k+2 < size*3:
                output_grid[w, k+1] = ' '
                output_grid[w, k+2] = ' '
            if direction == 1 and w-2 >= 0:
                output_grid[w-1, k] = ' '
                output_grid[w-2, k] = ' '
            j = j + 1
        i = i + 1
        j = 0
        
    return output_grid

## Pre-processing grid function.
#
# Avoid digging outside the maze external borders.
#
# @param grid Grid.
# #param size Length of three.
def preprocess_binomial(grid, size):

    first_row = grid[0]
    first_row[first_row == 1] = 0
    grid[0] = first_row
    for i in range(1, size):
        grid[i, size-1] = 1
    return grid

## Maze Binary Tree Int Pattern.
#
#  Generate matrix with 1-wall, 0-free path, 2-cell part of the shortest path. 
#
# @param maze matrix with the set values to build the scene.
def maze_int_binary_tree(maze):

    output_int = [[int(0) for col in row] for row in maze]
    z = 0
    w = 0
    starti = 0
    endi = 0
    for z in range(0, 24):
        for w in range(0, 24):
            if maze[z][w] == '#':
                output_int[z][w] = 1
                
    distance, prev_point = dijkstra(output_int, (22, 22))
    dij = find_shortest_path(prev_point, (19, 1))
    
    for elm in dij:
        output_int[elm[0]][elm[1]] = 2
    
    return output_int

## Dijkstra's algorithm function.
#
# Initialize graphs to track if a point is visited,
# current calculated distance from start to point,
# and previous point taken to get to current point.
#
# @param graph Graph.
# @param start_point Initial vertex/node for the algorithm.
def dijkstra(graph, start_point):
    
    visited = [[False for col in row] for row in graph]
    z = 0
    w = 0

    # Where there's a wall, visited is marked to limit the solution of shortest path.
    for z in range(0, 24):
        for w in range(0, 24):
            if graph[z][w] == 1:
                visited[z][w] = True

    distance = [[float('inf') for col in row] for row in graph]
    distance[start_point[0]][start_point[1]] = 0
    path = [[None for col in row] for row in graph]
    n, m = len(graph), len(graph[0])
    number_of_points, visited_count = n * m, 0
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    min_heap = []

    # Min_heap item format:
    # (pt's dist from start on this path, pt's row, pt's col).
    heapq.heappush(min_heap, (distance[start_point[0]][start_point[1]], start_point[0], start_point[1]))

    while visited_count < number_of_points:
        if min_heap != []:
            current_point = heapq.heappop(min_heap)
            distance_from_start, row, col = current_point
        for direction in directions:
            new_row, new_col = row + direction[0], col + direction[1]
            if -1 < new_row < n and -1 < new_col < m and not visited[new_row][new_col]:
                # Chooses if it's better take or not take the vertice to total path.
                dist_to_new_point = distance_from_start + graph[new_row][new_col]
                if dist_to_new_point < distance[new_row][new_col]:
                    distance[new_row][new_col] = dist_to_new_point
                    path[new_row][new_col] = (row, col)
                    heapq.heappush(min_heap, (dist_to_new_point, new_row, new_col))
        visited[row][col] = True
        visited_count += 1

    return distance, path

## Find Shortest Path function.
#
# Find the shortest path in a graph. 
# In that case, the path is only one, because all mazes have only 1 path from start to end.
#
# @param prev_point_graph Previous point of the graph.
# @param end_point End point of the graph.
def find_shortest_path(path_graph, end_point):

    shortest_path = []
    current_point = end_point
    i = 0
    while current_point is not None:
        i += 1
        shortest_path.append(current_point)
        current_point = path_graph[current_point[0]][current_point[1]]
    # Build path from end to start -> reverse way.
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
    model = np.matmul(Rx, S)
    model = np.matmul(Ry, model)
    loc = gl.glGetUniformLocation(program, "model")
    gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, model.transpose())
    
    projection = ut.matPerspective(np.radians(camera_posy+60.0), win_width/win_height, 0.1, 100.0)
    loc = gl.glGetUniformLocation(program, "projection")
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
        n = 1
        p = 0.5
        size = 8
        grid = np.random.binomial(n, p, size=(size,size))
        processed_grid = preprocess_binomial(grid, size)
        maze = maze_binary_tree(processed_grid, size)
        output_int = maze_int_binary_tree(maze)
        print("Maze generated by binary tree.")
        print("")
    elif algorithm == 2:
        x = 25
        y = 25
        maze = maze_kruskal(x, y)
        output_int = maze_int_kruskal(maze)
        print("Maze generated by kruskal.")
        print("")
    
    # Each cube will receive the color relationed with being or not part of
    # the free way of the maze and relationed with being part of the solution or not.
    offsety = -20
    offsetz = -20
    z = 0
    w = 0
    # Drawing a cube grid.
    for z in range(0, 24):
        for w in range(0, 24):
            # White -> the walls, Red -> the free path, Blue -> the solution.
            if(output_int[z][w] == 1):
                loc = gl.glGetUniformLocation(program, "objectColor")
                gl.glUniform3f(loc, 1.0, 1.0, 1.0)
            elif (output_int[z][w] == 0):
                loc = gl.glGetUniformLocation(program, "objectColor")
                gl.glUniform3f(loc, 1.5, 0.2, 0.2)
            else:
                loc = gl.glGetUniformLocation(program, "objectColor")
                gl.glUniform3f(loc, 0.2, 2.0, 0.8)
            
            # Draw one cube at a time.
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
def reshape(width, height):

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
        camera_posy += 0.5
    if key == b's' or key == b'S':
        camera_posy -= 0.5
    if key == b'a' or key == b'A':
        camera_posx -= 0.5
    if key == b'd' or key == b'D':
        camera_posx += 0.5
    
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
        # Coordinate       # Normal
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

    # Vertex buffer.
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

    gl.glEnable(gl.GL_DEPTH_TEST)

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
    glut.glutInitContextVersion(3, 3)
    glut.glutInitContextProfile(glut.GLUT_CORE_PROFILE)
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH)
    glut.glutInitWindowSize(win_width,win_height)
    glut.glutCreateWindow('Maze Generator Using Graph Algorithm')
    
    # Init data.
    initData()
    
    # Create shaders.
    initShaders()

    glut.glutReshapeFunc(reshape)
    glut.glutDisplayFunc(display)
    glut.glutKeyboardFunc(keyboard)

    glut.glutMainLoop()

if __name__ == '__main__':
    main()
