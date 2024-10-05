import numpy as np
import random
import matplotlib.pyplot as plt
from preprocessing import advanced_prepare_data

class LinearRegression:
    def initialize_params(self):
        bias = 0
        _ = random.gauss(mu=0.0, sigma=0.01) # w1
        _ = random.gauss(mu=0.0, sigma=0.01) # w2
        _ = random.gauss(mu=0.0, sigma=0.01) # w2 # return [bias, w1, w2, w3]
        
        return [bias, -0.01268850433497871, 0.004752496982185252, 0.0073796171538643845] 
        
    # Predict output by using y = x0*b + x1*w1 + x2*w2 + x3*w3
    def predict(self, x_features, weights):
        x_features = np.array(x_features)
        weights = np.array(weights)
        result = np.dot(x_features, weights)
        return result
    
    def compute_loss(self, y_hat, y):
        return (y_hat - y)**2
    
    def compute_gradient_w(self, x_feature, y, y_hat):
        x_feature = np.array(x_feature)
        dl_dweights = 2*x_feature*(y_hat - y)
        return dl_dweights
    
    def update_weight(self, weights, dl_dweights, lr):
        weights = np.array(weights)
        dl_dweights = np.array(dl_dweights)
        weights = weights - lr*dl_dweights
        return weights.tolist()
    
    def implement_linear_regression(self, x_feature, y_output, epoch_max = 50, lr = 1e-5):
        losses = []
        weights = self.initialize_params()
        N = len(y_output)
        for epoch in range(epoch_max):
            print("epoch", epoch)
            for i in range(N):
                # get a sample - row i
                features_i = x_feature[i]
                y = y_output[i]
                
                # compute output
                y_hat = self.predict(features_i, weights)
                
                # compute loss
                loss = self.compute_loss(y, y_hat)
                
                # compute gradient w1, w2, w3, b
                dl_dweights = self.compute_gradient_w(features_i, y, y_hat)

                # update parameters
                weights = self.update_weight(weights, dl_dweights, lr)

                # logging
                losses.append(loss)
                
        return weights, losses 
    
    def plot_result(self, losses, values=-1):
        plt.plot(losses[:values])
        plt.xlabel("#iteration")
        plt.ylabel("Loss")
        plt.show()
        
if __name__ == "__main__":
    # Testcases
    # Testcases 1
    X, y = advanced_prepare_data('./Module_4/Week_1/Data/advertising.csv') # Ensure this path is correct and points to your advertising.csv file
    regression = LinearRegression()
    W, L = regression.implement_linear_regression(X, y, epoch_max=100)
    regression.plot_result(losses=L, values=100)
    
    # Testcases 2
    W, L = regression.implement_linear_regression(X, y, epoch_max=50, lr =1e-5)
    # Print loss value at iteration 9999
    print(L[9999])
    
    
    
