import numpy as np
import matplotlib.pyplot as plt
from utils import load_iris_data

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.clusters = None
    
    def initialize_centroids(self, data):
        rng = np.random.default_rng(seed=42)
        self.centroids = data[rng.choice(data.shape[0], self.k, replace=False)]
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum(np.square(x1 - x2)))
    
    def assign_clusters(self, data):
        distances = np.array([[self.euclidean_distance(x, centroid) for centroid in self.centroids] for x in data])
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, data):
        return np.array([data[self.clusters == i].mean(axis=0) for i in range(self.k)])
    
    def fit(self, data):
        self.initialize_centroids(data)
        
        for i in range(self.max_iters):
            self.clusters = self.assign_clusters(data)
            
            self.plot_clusters(data, i)
            
            new_centroids = self.update_centroids(data)
            
            if np.all(self.centroids == new_centroids):
                break
            
            self.centroids = new_centroids
            
        self.plot_final_clusters(data)
        
    def plot_clusters(self, data, iteration):
        # Plot data
        plt.scatter(data[:, 0], data[:, 1], c=self.clusters, cmap='viridis',
                    marker='o', alpha=0.6)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x', s=300)
        plt.title(f"Iteration {iteration + 1}")
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.show()

    def plot_final_clusters(self, data):
        # Plot data
        plt.scatter(data[:, 0], data[:, 1], c=self.clusters, cmap='viridis',
                    marker='o', alpha=0.6)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x', s=300)
        plt.title("Final Clusters and Centroids")
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.show()
        
if __name__ == "__main__":
    data = load_iris_data()
    
    kmeans = KMeans()
    kmeans.fit(data)