import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
from OpenGL.GL.shaders import compileProgram,compileShader
import math
import time
from OpenGL.GLU import *

import glfw.GLFW as GLFW_CONSTANTS
from graphics import *
import glm  

# Vertex and Fragment Shaders with Matrix Transformation

def initialVectF(x,y,z):
   val=1,1,1
   if math.isnan(val[0]) or math.isnan(val[1]) or math.isnan(val[2]):
    val = 1,1,1 
   return val



def create_shader(vertex_filepath: str, fragment_filepath: str) -> int:
    with open(vertex_filepath, 'r') as f:
        vertex_src = f.read()

    with open(fragment_filepath, 'r') as f:
        fragment_src = f.read()

    vertex_shader = compileShader(vertex_src, GL_VERTEX_SHADER)
    fragment_shader = compileShader(fragment_src, GL_FRAGMENT_SHADER)

    shader = glCreateProgram()
    glAttachShader(shader, vertex_shader)
    glAttachShader(shader, fragment_shader)
    glLinkProgram(shader)

    # Check for errors
    if not glGetProgramiv(shader, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(shader))

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader




def create_projection_matrix(fov, aspect_ratio, near, far):
    f = 1.0 / np.tan(np.radians(fov) / 2.0)
    projection_matrix = np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)
    return projection_matrix

def create_view_matrix_with_rotation(eye, target, up, rotation_angles):
    """
    Creates a view matrix with rotation applied around the x, y, and z axes.

    :param eye: Camera position (vec3)
    :param target: Look-at point (vec3)
    :param up: Up vector (vec3)
    :param rotation_angles: Rotation angles (theta_x, theta_y, theta_z) in degrees
    :return: The view matrix (4x4 numpy array)
    """
    # Standard view matrix calculation
    f = target - eye
    f = f / np.linalg.norm(f)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    view_matrix = np.eye(4, dtype=np.float32)
    view_matrix[0, :3] = s
    view_matrix[1, :3] = u
    view_matrix[2, :3] = -f
    view_matrix[:3, 3] = -np.dot(view_matrix[:3, :3], eye)

    # Rotation matrices
    theta_x, theta_y, theta_z = np.radians(rotation_angles)  # Convert degrees to radians

    rotation_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x), 0],
        [0, np.sin(theta_x), np.cos(theta_x), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    rotation_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    rotation_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0, 0],
        [np.sin(theta_z), np.cos(theta_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # Combine rotations: Rz * Ry * Rx
    rotation_matrix = np.matmul(np.matmul(rotation_z, rotation_y), rotation_x)

    # Apply rotation to the view matrix
    rotated_view_matrix = np.matmul( view_matrix,rotation_matrix)

    return rotated_view_matrix

class App:


    def __init__(self):
        """ Initialise the program """

        self.initialize_glfw()
        self.initialize_opengl()





 


    def initialize_glfw(self) -> None:
        """
            Initialize all glfw related stuff. Make a window, basically.
        """
        glfw.init()
        glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MAJOR,3)
        glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MINOR,3)
        glfw.window_hint(
            GLFW_CONSTANTS.GLFW_OPENGL_PROFILE,
            GLFW_CONSTANTS.GLFW_OPENGL_CORE_PROFILE
        )
        glfw.window_hint(
            GLFW_CONSTANTS.GLFW_OPENGL_FORWARD_COMPAT,
            GLFW_CONSTANTS.GLFW_TRUE
        )
        self.window = glfw.create_window(
            800, 600, "Title", None, None)
        glfw.make_context_current(self.window)
   
    def initialize_opengl(self) -> None:
        """
            Initialize any opengl related stuff.
        """
        #glClearColor(0.1, 0.2, 0.2, 1)

        #self.triangle_buffers, self.triangle_vao = mesh_factory.build_triangle_mesh()
        self.rectangle = Rectangle()
        self.shader = create_shader("shaders/vertex.txt", "shaders/fragment.txt")
        self.shader2 = create_shader("shaders/vertex2.txt", "shaders/fragment2.txt")
        self.vectors = [
                (np.array([i, j, k]), np.array([i/5, j/5, k/5]))
                for i in np.arange(-5, 6)
                for j in np.arange(-5, 6)
                for k in np.arange(-5, 6)
            ]


    def run(self):
        """ Run the app """

        t=0
        while not glfw.window_should_close(self.window):
     
            if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_ESCAPE) \
                == GLFW_CONSTANTS.GLFW_PRESS:
                break
            glfw.poll_events()

        
            #draw triangle
       
            
            
            eye = np.array([0.0, 0.0, 5.0], dtype=np.float32)  # Camera position
            target = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Look-at point
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            view_matrix = create_view_matrix_with_rotation(eye, target, up, (t, t, t))
            projection_matrix = create_projection_matrix(45.0, 800.0 / 600.0, 0.1, 100.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            translation = np.array([
                [1, 0, 0, .1],  # Translate 2 units to the right
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
      
            t+=1
           # glUniformMatrix4fv(glGetUniformLocation(self.shader2, "model"), 1, GL_FALSE, translation)
         
            draw_vector_field(self.vectors,view_matrix,projection_matrix,self.shader)

            glfw.swap_buffers(self.window)

    def quit(self):
        """ cleanup the app, run exit code """

        #glDeleteBuffers(len(self.triangle_buffers), self.triangle_buffers)
        glDeleteBuffers(3, (self.triangle_vbo, self.quad_ebo, self.quad_vbo))
        glDeleteVertexArrays(2, (self.triangle_vao, self.quad_vao))
        glDeleteProgram(self.shader)
        glfw.destroy_window(self.window)
        glfw.terminate()

my_app = App()
my_app.run()
my_app.quit()

