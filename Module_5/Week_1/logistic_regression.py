import numpy as np
from config import ModelConfig, DatasetConfig

class LogisticRegression:
    def __init__(self, x_train, y_train, x_val, y_val, model_name="Titanic"):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.config = ModelConfig.model_type.get(model_name)
            
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
    
    def _update_theta(self, theta, gradient, lr):
        return theta - lr * gradient
    
    def _compute_accuracy(self, x, y, theta):
        y_hat = self._predict(x, theta).round()
        acc = (y_hat == y).mean()
        return acc
    
    def train(self):
        train_accs = []
        train_losses = []
        val_accs = []
        val_losses = []
        
        theta = self._initialize_weights(self.x_train.shape[1])
        learning_rate = self.config["LEARNING_RATE"]
        epochs = self.config["EPOCHS"]
        batch_size = self.config["BATCH_SIZE"]
        
        for epoch in range(epochs):
            train_batch_losses = []
            train_batch_accs = []
            val_batch_losses = []
            val_batch_accs = []
            
            for i in range(0, self.x_train.shape[0], batch_size):
                x_i = self.x_train[i:i+batch_size]
                y_i = self.y_train[i:i+batch_size]
                
                y_hat = self._predict(x_i, theta)
                
                train_loss = self._compute_loss(y_hat, y_i)
                
                gradient = self._compute_gradient(x_i, y_i, y_hat)
                
                theta = self._update_theta(theta, gradient, learning_rate)
                
                train_batch_losses.append(train_loss)
                
                train_acc = self._compute_accuracy(self.x_train, self.y_train, theta)
                train_batch_accs.append(train_acc)
                
                y_val_hat = self._predict(self.x_val, theta)
                val_loss = self._compute_loss(y_val_hat, self.y_val)
                val_batch_losses.append(val_loss)
                
                val_acc = self._compute_accuracy(self.x_val, self.y_val, theta)
                val_batch_accs.append(val_acc) 
                
            train_batch_loss = sum(train_batch_losses) / len(train_batch_losses)
            val_batch_loss = sum(val_batch_losses) / len(val_batch_losses)
            train_batch_acc = sum(train_batch_accs) / len(train_batch_accs)
            val_batch_acc = sum(val_batch_accs) / len(val_batch_accs)
            
            train_losses.append(train_batch_loss)
            val_losses.append(val_batch_loss)
            train_accs.append(train_batch_acc)
            val_accs.append(val_batch_acc)
            
            print(f'\nEPOCH {epoch + 1}:\tTraining loss: {train_batch_loss :.3f}\
                  \tValidation loss: {val_batch_loss:.3f}')
        return train_losses, val_losses, train_accs, val_accs, theta

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