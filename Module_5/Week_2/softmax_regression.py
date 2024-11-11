import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import DatasetConfig
from Module_5.Week_1.base_regression import BaseRegression

class SoftmaxRegression(BaseRegression):
    def __init__(self, x_train, y_train, x_val, y_val, n_classes, model_name="Titanic"):
        super().__init__(x_train, y_train, x_val, y_val, model_name)
        self.n_classes = n_classes
        
    def _initialize_weights(self, feature_count, random_state=DatasetConfig.RANDOM_SEED):
        rng = np.random.default_rng(random_state)
        return rng.uniform(size=(feature_count, self.n_classes))

    def _softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1)[:, None]
    
    def _predict(self, x, theta):
        z = np.dot(x, theta)
        y_hat = self._softmax(z)
        return y_hat
    
    def _compute_loss(self, y_hat, y):
        n = y.size
        return (-1/n) * np.sum(y * np.log(y_hat))
    
    def _compute_gradient(self, x, y, y_hat):
        n = y.size
        return np.dot(x.T, (y_hat - y)) / n
    
    def _compute_accuracy(self, x, y, theta):
        y_hat = self._predict(x, theta)
        acc = (np.argmax(y_hat, axis=1) == np.argmax(y, axis=1)).mean()
        return acc
    
if __name__ == "__main__":
    regression = SoftmaxRegression(None, None, None, None, None)
    # Testcases
    # Check loss
    y = np.array([1, 0, 0, 0])
    y_hat =  np.array([0.4, 0.15, 0.05, 0.4])
    print(f"Loss: {regression._compute_loss(y_hat, y) * y.size}") # Loss: 0.916290731874155
    
    # Check softmax
    z = (-1, -2, 3, 2)
    def softmax(z):
        """Define a new softmax for z"""
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()
    print(f"Prob: P = {np.round(softmax(z), 3)}") # Prob: P = [0.013 0.005 0.718 0.264]
    
    # Check Accuracy
    def compute_accuracy(y_hat, y):
        """Define a new accuracy function get defined y_hat"""
        acc = (y_hat == y).mean()
        return acc
    y = np.array([0, 0, 3, 2, 1, 2, 2, 1])
    y_hat =  np.array([0, 1, 3, 2, 0, 2, 1, 2])
    print(f"Accuracy: {compute_accuracy(y_hat, y)}") # Accuracy: 0.5