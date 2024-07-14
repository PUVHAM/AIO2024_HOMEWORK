import numpy as np

def compute_cosine(v1, v2):
    if not isinstance(v1, np.ndarray):
        v1 = np.array(v1)
    if not isinstance(v2, np.ndarray):
        v2 = np.array(v2)
        
    cos_sim = (v1 @ v2) / (np.sum(np.square(v1)) * np.sum(np.square(v2))) ** 0.5
    return cos_sim

print(f"Cosine similarity: {round(compute_cosine([1, 2, 3, 4],[1, 0, 3, 0]), 3)}")