import numpy as np

def compute_eigenvalues_eigenvectors(matrix):
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
        
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    return eigenvalues, eigenvectors

eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors([[0.9, 0.2],[0.1, 0.8]])
print(f"Eigenvalues: {eigenvalues} \nEigenvector: \n{eigenvectors}")

