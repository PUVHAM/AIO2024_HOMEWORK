import numpy as np
from config import ModelConfig, DatasetConfig

class BaseRegression:
    def __init__(self, x_train, y_train, x_val, y_val, model_name="Titanic"):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.config = ModelConfig.model_type.get(model_name)
            
    def _initialize_weights(self, feature_count, random_state=DatasetConfig.RANDOM_SEED):
        rng = np.random.default_rng(random_state)
        return rng.uniform(size=feature_count)
    
    def _compute_loss(self, y_hat, y):
        raise NotImplementedError
    
    def _compute_gradient(self, x, y, y_hat):
        raise NotImplementedError
    
    def _update_theta(self, theta, gradient, lr):
        return theta - lr * gradient
    
    def _compute_accuracy(self, x, y, theta):
        y_hat = self._predict(x, theta)
        acc = (np.argmax(y_hat, axis=1) == np.argmax(y, axis=1)).mean() if self.n_classes > 2 else (y_hat.round() == y).mean()
        return acc

    def _predict(self, x, theta):
        raise NotImplementedError

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
            
            print(f'\nEPOCH {epoch + 1}:\tTraining loss: {train_batch_loss :.3f}\tValidation loss: {val_batch_loss:.3f}')
        
        return train_losses, val_losses, train_accs, val_accs, theta