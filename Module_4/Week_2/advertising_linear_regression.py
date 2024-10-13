import numpy as np
import matplotlib.pyplot as plt

# Normalize input data by using mean normalizaton
def mean_normalization(x):
    N = len(x)
    maximum = np.max(x)
    minimum = np.min(x)
    avg = np.mean(x)
    x = (x-avg) / (maximum - minimum) 
    x_b = np.c_[np.ones((N, 1)), x]
    return x_b, maximum, minimum, avg

class LinearRegression:    
    def predict(self, x_features, theta):
        return np.dot(x_features, theta)
    
    def compute_loss(self, y_hat, y):
        return (y_hat - y)**2 / 2
    
    def compute_loss_gradient(self, y_hat, y):
        return y_hat - y
    
    def compute_gradient(self, x_feature, g_li):
        return x_feature.T.dot(g_li)
    
    def update_weight(self, theta, gradient, lr):
        return theta - lr*gradient
        
    def stochastic_gradient_descent(self, x_b, y, n_epochs=50, learning_rate=0.00001):
    
        thetas = np.asarray([[1.16270837], [-0.81960489], [1.39501033], [0.29763545]]) # np.random.randn(4, 1) # use this code for real application
        
        thetas_path = [thetas]
        losses = []
        
        for _ in range(n_epochs):
            for i in range(N):
                # select random number in N 
                # random_index = np.random.randint(N) # In real application, you should use this code
                random_index = i # This code is used for this assignment only
                
                xi = x_b[random_index:random_index+1]
                yi = y[random_index:random_index+1]
                
                # Compute output
                y_hat = self.predict(xi, thetas)
                
                # Compute loss li
                li = self.compute_loss(y_hat, yi)
                
                # Compute gradient for loss
                g_li = self.compute_loss_gradient(y_hat, yi)
                
                # Compute gradient
                grad = self.compute_gradient(xi, g_li)
                
                # Update theta
                thetas = self.update_weight(thetas, grad, learning_rate)
                
                # logging
                thetas_path.append(thetas)
                losses.append(li[0][0])
        return thetas_path, losses
    
    def mini_batch_gradient_descent(self, x_b, y, n_epochs=50, minibatch_size = 20, learning_rate=0.00001):
        thetas = np.asarray([[1.16270837], [-0.81960489], [1.39501033], [0.29763545]]) # np.random.randn(4, 1) # use this code for real application
        
        thetas_path = [thetas]
        losses = []
        
        for _ in range(n_epochs):
            shuffled_indices = np.asarray([21, 144, 17, 107, 37, 115, 167, 31, 3, 132, 179, 155, 36, 191, 182, 
                                170, 27, 35, 162, 25, 28, 73, 172, 152, 102, 16, 185, 11, 1, 34, 
                                177, 29, 96, 22, 76, 196, 6, 128, 114, 117, 111, 43, 57, 126, 165, 
                                78, 151, 104, 110, 53, 181, 113, 173, 75, 23, 161, 85, 94, 18, 148, 
                                190, 169, 149, 79, 138, 20, 108, 137, 93, 192, 198, 153, 4, 45, 164, 
                                26, 8, 131, 77, 80, 130, 127, 125, 61, 10, 175, 143, 87, 33, 50, 54, 
                                97, 9, 84, 188, 139, 195, 72, 64, 194, 44, 109, 112, 60, 86, 90, 140, 
                                171, 59, 199, 105, 41, 147, 92, 52, 124, 71, 197, 163, 98, 189, 103, 
                                51, 39, 180, 74, 145, 118, 38, 47, 174, 100, 184, 183, 160, 69, 91, 
                                82, 42, 89, 81, 186, 136, 63, 157, 46, 67, 129, 120, 116, 32, 19, 
                                187, 70, 141, 146, 15, 58, 119, 12, 95, 0, 40, 83, 24, 168, 150, 178, 
                                49, 159, 7, 193, 48, 30, 14, 121, 5, 142, 65, 176, 101, 55, 133, 13, 
                                106, 66, 99, 68, 135, 158, 88, 62, 166, 156, 2, 134, 56, 123, 122, 
                                154]) # np.random.permutation(N) # use this code for real application
            
            x_b_shuffled = x_b[shuffled_indices]
            y_shuffled = y[shuffled_indices]
            
            for i in range(0, N, minibatch_size):
                xi = x_b_shuffled[i:i+minibatch_size]
                yi = y_shuffled[i:i+minibatch_size]
                
                # Compute output
                y_hat = self.predict(xi, thetas)
                
                # Compute loss li
                loss = self.compute_loss(y_hat, yi)
                
                # Compute gradient for loss
                g_li = self.compute_loss_gradient(y_hat, yi)/minibatch_size
                
                # Compute gradient
                grad = self.compute_gradient(xi, g_li)
                
                # Update theta
                thetas = self.update_weight(thetas, grad, learning_rate)
                
                # logging
                thetas_path.append(thetas)
                loss_mean = np.sum(loss)/minibatch_size
                losses.append(loss_mean)
        return thetas_path, losses
    
    def batch_gradient_descent(self, x_b, y, n_epochs=50, learning_rate=0.00001):
        thetas = np.asarray([[1.16270837], [-0.81960489], [1.39501033], [0.29763545]]) # np.random.randn(4, 1) # use this code for real application
        
        thetas_path = [thetas]
        losses = []
        
        for _ in range(n_epochs):            
            # Compute output
            y_hat = self.predict(x_b, thetas)
            
            # Compute loss li
            loss = 2*self.compute_loss(y_hat, y)
            
            # Compute gradient for loss
            g_li = 2*self.compute_loss_gradient(y_hat, y)/N
            
            # Compute gradient
            grad = self.compute_gradient(x_b, g_li)
            
            # Update theta
            thetas = self.update_weight(thetas, grad, learning_rate)
            
            # logging
            thetas_path.append(thetas)
            loss_mean = np.sum(loss)/N
            losses.append(loss_mean)
        return thetas_path, losses
        
if __name__ == "__main__":
    # dataset
    data = np.genfromtxt('./Module_4/Week_2/Data/advertising.csv', delimiter=',', skip_header=1) # Ensure this path is correct and points to your advertising.csv file
    N = data.shape[0]
    X = data[:, :3]
    y = data[:, 3:]
    
    X_b, maxi, mini, avg = mean_normalization(X)
    
    regression = LinearRegression()
    # Stochastic Gradient Descent
    sgd_theta, losses = regression.stochastic_gradient_descent(X_b, y, n_epochs=50, learning_rate=0.01)

    x_axis = list(range(500))
    plt.plot(x_axis, losses[:500], color="r")
    plt.show()

    sgd_theta, losses = regression.stochastic_gradient_descent(X_b, y, n_epochs=1, learning_rate=0.01)
    print(round(sum(losses), 2)) # 6754.64
    
    # Mini-batch Gradient Descent
    mbgd_thetas, losses = regression.mini_batch_gradient_descent(X_b, y, n_epochs=50,  minibatch_size = 20, learning_rate=0.01)

    x_axis = list(range(200))
    plt.plot(x_axis,losses[:200], color="r")
    plt.show()
    print(round(sum(losses), 2)) # 8865.65
    
    # Batch Gradient Descent
    bgd_thetas, losses = regression.batch_gradient_descent(X_b, y, n_epochs=100, learning_rate=0.01)

    x_axis = list(range(100))
    plt.plot(x_axis,losses[:100], color="r")
    plt.show()
    print(round(sum(losses), 2)) # 6716.46