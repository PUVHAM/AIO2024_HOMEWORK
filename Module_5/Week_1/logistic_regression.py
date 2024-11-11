import numpy as np
from config import DatasetConfig
from base_regression import BaseRegression

class LogisticRegression(BaseRegression):
    def __init__(self, x_train, y_train, x_val, y_val, model_name="Titanic"):
        super().__init__(x_train, y_train, x_val, y_val, model_name)

    def _initialize_weights(self, feature_count, random_state=DatasetConfig.RANDOM_SEED):
        rng = np.random.default_rng(random_state)
        return rng.uniform(size=feature_count)
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _predict(self, x, theta):
        dot_product = np.dot(x, theta)
        y_hat = self._sigmoid(dot_product)
        return y_hat
    
    def _compute_loss(self, y_hat, y):
        y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)
        return (-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)).mean()
    
    def _compute_gradient(self, x, y, y_hat):
        return np.dot(x.T, (y_hat - y)) / y.size
    
    def _compute_accuracy(self, x, y, theta):
        y_hat = self._predict(x, theta).round()
        acc = (y_hat == y).mean()
        return acc

if __name__ == "__main__":
    regression = LogisticRegression(None, None, None, None)
    # Testcases
    # Check prediction
    X = [[22.3, -1.5, 1.1, 1]]
    theta = [0.1, -0.15, 0.3, -0.2]
    print(f"Prediction: {regression._predict(X, theta)[0]}") # Prediction: 0.9298899437785819
    
    # Check loss
    y = np.array([1, 0, 0, 1])
    y_hat =  np.array([0.8, 0.75, 0.3, 0.95])
    print(f"Loss: {regression._compute_loss(y_hat, y)}") # Loss: 0.5043515376900958
    
    # Check gradient 
    X = np.array([[1, 2], [2, 1], [1, 1], [2, 2]])
    y_true = np.array([0, 1, 0, 1])
    y_pred = [0.25, 0.75, 0.4, 0.8]
    print(f"Gradient 1: {regression._compute_gradient(X, y_true, y_pred)}") # Gradient 1: [-0.0625  0.0625]
    
    X = np.array([[1, 3], [2, 1], [3, 2], [1, 2]])
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([0.7, 0.4, 0.6, 0.85])
    print(f"Gradient 2: {regression._compute_gradient(X, y_true, y_pred)}") # Gradient 2: [-0.2125 -0.4   ]

    # Check accuracy
    def compute_accuracy(y_true, y_pred):
        """Define a new function get defined y_pred"""
        y_pred_rounded = np.round(y_pred)
        accuracy = np.mean(y_true == y_pred_rounded)
        return accuracy
    y_true = [1, 0, 1, 1] 
    y_pred = [0.85, 0.35, 0.9, 0.75]
    print(f"Accuracy: {compute_accuracy(y_true, y_pred)}") # Accuracy: 1.0 