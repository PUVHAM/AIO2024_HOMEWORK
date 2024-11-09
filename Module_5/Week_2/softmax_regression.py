import numpy as np
from config import ModelConfig, DatasetConfig

class SoftmaxRegression:
    def __init__(self, x_train, y_train, x_val, y_val, n_classes, model_name="Titanic"):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self._model_name = model_name
        self.n_classes = n_classes
        self.config = ModelConfig.model_type.get(self._model_name)
            
    def _initialize_weights(self, n_features, random_state=DatasetConfig.RANDOM_SEED):
        rng = np.random.default_rng(random_state)
        return rng.uniform(size=(n_features, self.n_classes))
        
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
    
    def _update_theta(self, theta, gradient, lr):
        return theta - lr * gradient
    
    def _compute_accuracy(self, x, y, theta):
        y_hat = self._predict(x, theta)
        acc = (np.argmax(y_hat, axis=1) == np.argmax(y, axis=1)).mean()
        return acc
    
    def train(self):
        train_accs = []
        train_losses = []
        val_accs = []
        val_losses = []
        

        learning_rate = self.config["LEARNING_RATE"]
        epochs = self.config["EPOCHS"]
        batch_size = self.config["BATCH_SIZE"]
        n_features = self.x_train.shape[1]
        theta = self._initialize_weights(n_features)
        
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