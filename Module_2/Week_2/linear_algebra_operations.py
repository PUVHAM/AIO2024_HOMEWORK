import numpy as np

def compute_vector_length(vector):
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector)
    
    sum_vector = np.sum(np.square(vector))
    len_of_vector = sum_vector ** 0.5
    
    return len_of_vector

def compute_dot_product(vector1, vector2):
    if not isinstance(vector1, np.ndarray):
        vector1 = np.array(vector1)
    if not isinstance(vector2, np.ndarray):
        vector2 = np.array(vector2)
        
    result = np.dot(vector1, vector2)
        
    return result

def matrix_multi_vector(matrix, vector):
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector)
        
    result = matrix @ vector
    
    return result

def matrix_multi_matrix(matrix1, matrix2):
    if not isinstance(matrix1, np.ndarray):
        matrix1 = np.array(matrix1)
    if not isinstance(matrix2, np.ndarray):
        matrix2 = np.array(matrix2)
      
    len_of_vector = matrix1 @ matrix2
  
    return len_of_vector

def inverse_matrix(vector):
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector)
        
    determinant = np.linalg.det(vector)
    
    if determinant == 0:
        raise ValueError("Matrix cannot be inverted")
    
    result = np.linalg.inv(vector)
    
    return result

# Testcases
vector = np.array([-2, 4, 9, 21])
result = compute_vector_length([vector])
print("Length of a vector:", round(result, 2))

v1 = np.array([0, 1, -1, 2])
v2 = np.array([2, 5, 1, 0])
result = compute_dot_product(v1, v2)
print(f"Dot product: {round(result,2)}")

m = np.array([[-1, 1, 1], [0, -4, 9]])
v = np.array([0, 2, 1])
print(f"Multiplying a vector by a matrix: {matrix_multi_matrix(m, v)}")

m1 = np.array([[0, 1, 2], [2, -3, 1]])
m2 = np.array([[1, -3], [6, 1], [0, -1]])
print(f"Multiplying a matrix by a matrix: {matrix_multi_matrix(m1, m2)}")

m1 = np.array([[-2, 6], [8, -4]])
print(f"Inverse matrix: {inverse_matrix(m1)}")