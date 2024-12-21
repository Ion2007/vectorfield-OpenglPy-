import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import math
import time
from OpenGL.GLU import *
from OpenGL.GL.shaders import compileProgram,compileShader

class Triangle:
    """
        Yep, it's a triangle.
    """


    def __init__(self, tip, b1, b2):
        """
            Initialize a triangle.
        """

        # x, y, z, r, g, b
        vertices = (
            tip[0], tip[1], tip[2], 1.0, 1.0, 1.0,
             b1[0], b1[1], b1[2], 1.0, 1.0, 1.0,
             b2[0], b2[1], b2[2], 0.0, 0.0, 1.0
        )
        vertices = np.array(vertices, dtype=np.float32)

        self.vertex_count = 3

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
       
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
   
    def arm_for_drawing(self) -> None:
        """
            Arm the triangle for drawing.
        """
        glBindVertexArray(self.vao)
   
    def draw(self) -> None:
        """
            Draw the triangle.
        """

        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

    def destroy(self) -> None:
        """
            Free any allocated memory.
        """
       
        glDeleteVertexArrays(1,(self.vao,))
        glDeleteBuffers(1,(self.vbo,))

class Line:
    """
        Yep, it's a triangle.
    """


    def __init__(self, x,y,z,dx,dy,dz):
        """
            Initialize a triangle.
        """
   
        # x, y, z, r, g, b

        vertices = (
            x, y, z, 0.0, 0.0, 1.0,
             dx, dy, dz, 0.0, 0.0, 1.0
        )
        vertices = np.array(vertices, dtype=np.float32)

        self.vertex_count = 2

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
       
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
   
    def arm_for_drawing(self) -> None:
        """
            Arm the triangle for drawing.
        """
        glBindVertexArray(self.vao)
   
    def draw(self) -> None:
        """
            Draw the triangle.
        """

        glDrawArrays(GL_LINES, 0, self.vertex_count)

    def destroy(self) -> None:
        """
            Free any allocated memory.
        """
       
        glDeleteVertexArrays(1,(self.vao,))
        glDeleteBuffers(1,(self.vbo,))
def draw_vector(x, y, z, dx, dy, dz):
    # Calculate vector magnitude
    magnitude = math.sqrt(dx**2 + dy**2 + dz**2)
    if magnitude == 0:
        return  # Skip drawing if the vector has zero length

    # Normalize direction for consistent triangle size
    ux, uy, uz = dx / magnitude, dy / magnitude, dz / magnitude

    # Draw the line based on vector magnitude

    line=Line(x, y, z,x + dx, y + dy, z + dz)
   
    line.arm_for_drawing()
    line.draw()
    line.destroy()
 

    # Arrowhead (triangle) size and position
    arrow_size = 0.1  # Constant size of the arrowhead
    arrow_length = 0.9  # Position arrowhead close to the end of the vector

    # Arrowhead base position
    tx, ty, tz = x + dx * arrow_length, y + dy * arrow_length, z + dz * arrow_length

    # Perpendicular vector for arrowhead orientation (rotation based on cross product)
    perp = np.cross([ux, uy, uz], [0, 1, 0])
    if np.linalg.norm(perp) < 1e-6:  # Handle parallel vectors
        perp = np.cross([ux, uy, uz], [0, 0, 1])
    perp = perp / np.linalg.norm(perp) * arrow_size

    # Rotate the perpendicular vector to create a fixed-size triangle around the endpoint
    arrow_tip = [x + dx, y + dy, z + dz]
    arrow_base1 = [tx + perp[0], ty + perp[1], tz + perp[2]]
    arrow_base2 = [tx - perp[0], ty - perp[1], tz - perp[2]]

    # Draw the triangle
    version = glGetString(GL_VERSION)
   # print(f"OpenGL Version: {version}")
    if not version:
        raise Exception("Failed to create OpenGL context!")
    triangle=Triangle(arrow_tip,arrow_base1,arrow_base2)

    triangle.arm_for_drawing()
    triangle.draw()
 

class Rectangle:
    """
        Yep, it's a triangle.
    """


    def __init__(self):
        """
            Initialize a triangle.
        """
       
        # x, y, z, r, g, b
        vertices = (
            -0.5, -0.5, -10.0, 1.0, 0.0, 0.0,
             0.5, -0.5, -10.0, 0.0, 1.0, 0.0,
             0.5,  0.5, -10.0, 0.0, 0.0, 1.0,
            -0.5,  0.5, -10.0, 0.0, 0.0, 1.0,

            -0.5, -0.5, 0.0,   1.0, 0.0, 0.0,
             0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
            0.5,  0.5, 0.0, 0.0, 0.0, 1.0,
              -0.5,  0.5, 0.0,  0.0, 0.0, 1.0
        )
        indices = np.array([0, 1, 3, 3, 1, 2, 4,5,7,7,5,6], dtype=np.uint32)
        vertices = np.array(vertices, dtype=np.float32)

        self.vertex_count = 3

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
       
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
       
        self.ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    def arm_for_drawing(self) -> None:
        """
            Arm the triangle for drawing.
        """
        glBindVertexArray(self.vao)
   
    def draw(self) -> None:
        """
            Draw the triangle.
        """

        glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, ctypes.c_void_p(None))

    def destroy(self) -> None:
        """
            Free any allocated memory.
        """
       
        glDeleteVertexArrays(1,(self.vao,))
        glDeleteBuffers(1,(self.vbo,))

def generate_model_matrix(start, vector):
    """
    Generates a model matrix for translating, rotating, and scaling the vector.
    :param start: The starting position of the vector (vec3)
    :param vector: The direction and magnitude of the vector (vec3)
    :return: A 4x4 model matrix
    """
    # Calculate magnitude for scaling
    if vector[1]==0: vector[1]=.00001
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return np.eye(4, dtype=np.float32)  # Identity matrix for zero-length vectors

    # Normalize the vector
    normalized_vector = vector / magnitude
  

    # Default unit vector direction
    default_vector = np.array([0, 1, 0])

    # Compute the rotation axis and angle
    rotation_axis = np.cross(default_vector, normalized_vector)
    axis_length = np.linalg.norm(rotation_axis)

    if axis_length == 0:
        # Parallel or anti-parallel vectors
        rotation_axis = np.array([1, 0, 0], dtype=np.float32)  # Arbitrary axis
        dot_product = np.dot(default_vector, normalized_vector)
        angle = np.pi if dot_product < 0 else 0
    else:
        # Normalize the rotation axis
        rotation_axis /= axis_length
        # Calculate the rotation angle
        dot_product = np.dot(default_vector, normalized_vector)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))




    # Generate rotation matrix using axis-angle representation
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = rotation_axis

    rotation_matrix = np.array([
        [t * x * x + c,     t * x * y - s * z, t * x * z + s * y, 0],
        [t * x * y + s * z, t * y * y + c,     t * y * z - s * x, 0],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c,     0],
        [0,                 0,                 0,                 1]
    ], dtype=np.float32)



    # Create the translation matrix
    translation_matrix = np.eye(4, dtype=np.float32)
    translation_matrix[:3, 3] = start



    # Scaling matrix
    scaling_matrix = np.eye(4, dtype=np.float32)
    scaling_matrix[0, 0] = magnitude
    scaling_matrix[1, 1] = magnitude
    scaling_matrix[2, 2] = magnitude



    # Correct order: scale -> rotate -> translate
    model_matrix =np.matmul(rotation_matrix, scaling_matrix)
    model_matrix[0][3]+=start[0]
    model_matrix[1][3]+=start[1]
    model_matrix[2][3]+=start[2]



    return model_matrix
def draw_vector_field(vectors, view_matrix, projection_matrix, shader):
    """
    Draw a vector field using batched rendering and instanced draw calls.

    :param vectors: List of (start, vector) pairs.
    :param view_matrix: 4x4 view matrix.
    :param projection_matrix: 4x4 projection matrix.
    :param shader: Shader program ID.
    """
    glUseProgram(shader)

    # Vertex data for a single vector: line + triangle
    vertices = np.array([
        [0.0, 0.0, 0.0],  # Start of line
        [0.0, 1.0, 0.0],  # End of line
        [0.0, 1.5, 0.0],  # Triangle tip
        [0.1, 1.0, 0.0],  # Triangle right
        [-0.1, 1.0, 0.0]  # Triangle left
    ], dtype=np.float32).flatten()

    colors = np.array([
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0],  # Yellow
        [0.0, 1.0, 1.0]   # Cyan
    ], dtype=np.float32).flatten()

    # Precompute model matrices
    model_matrices = np.array([generate_model_matrix(start, vector) for start, vector in vectors], dtype=np.float32)

    # Create VAO and VBOs
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    # VBO for vertex positions
    vbo_vertices = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # VBO for vertex colors
    vbo_colors = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_colors)
    glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)

    # VBO for model matrices (instanced data)
   # VBO for model matrices (instanced data)
    vbo_instances = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_instances)

    # Transpose the matrices to match OpenGL's column-major format
    model_matrices_transposed = model_matrices.transpose(0, 2, 1).reshape(-1, 16)  # Swap axes 1 and 2, flatten the data
    glBufferData(GL_ARRAY_BUFFER, model_matrices_transposed.nbytes, model_matrices_transposed, GL_STATIC_DRAW)

    # Link each column (row in NumPy) of the model matrix as a separate attribute
    for i in range(4):
        glVertexAttribPointer(2 + i, 4, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), ctypes.c_void_p(i * 4 * sizeof(GLfloat)))
        glEnableVertexAttribArray(2 + i)
        glVertexAttribDivisor(2 + i, 1)  # Mark as instanced data

    # Set view and projection matrices
    view_location = glGetUniformLocation(shader, "view")
    projection_location = glGetUniformLocation(shader, "projection")
    glUniformMatrix4fv(view_location, 1, GL_TRUE, view_matrix)
    glUniformMatrix4fv(projection_location, 1, GL_TRUE, projection_matrix)

    # Draw instanced vectors
    glBindVertexArray(vao)
    glDrawArraysInstanced(GL_LINES, 0, 2, len(vectors))  # Line part
    glDrawArraysInstanced(GL_TRIANGLES, 2, 3, len(vectors))  # Triangle part

    # Cleanup
    glBindVertexArray(0)
    glDeleteBuffers(1, [vbo_vertices])
    glDeleteBuffers(1, [vbo_colors])
    glDeleteBuffers(1, [vbo_instances])
    glDeleteVertexArrays(1, [vao])