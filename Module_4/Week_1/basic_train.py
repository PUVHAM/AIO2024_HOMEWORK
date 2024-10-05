import numpy as np
import matplotlib.pyplot as plt
from preprocessing import prepare_data

class OneSampleLR:
    def initialize_params(self):
        w1, w2, w3, b = (0.016992259082509283, 0.0070783670518262355, -0.002307860847821344, 0)
        return w1, w2, w3, b
    
    def predict(self, x1, x2, x3, w1, w2, w3, b):
        y_pred = w1*x1 + w2*x2 + w3*x3 + b
        return y_pred
    
    def compute_loss(self, y_hat, y, loss_type):
        if loss_type == 'mse':
            loss = np.power(y_hat - y, 2)
        elif loss_type == 'mae':
            loss = np.abs(y_hat - y)
        else:
            raise ValueError(f"Unsupported loss type '{loss_type}'. Supported types are 'mse' and 'mae'.")
        return loss
    
    def compute_gradient_wi(self, xi, y, y_hat):
        dl_dwi = 2*xi*(y_hat - y)
        return dl_dwi
    
    def compute_gradient_b(self, y, y_hat):
        dl_db = 2*(y_hat - y)
        return dl_db
    
    def update_weight_wi(self, wi, dl_dwi, lr):
        wi = wi - lr*dl_dwi
        return wi
    
    def update_weight_b(self, b, dl_db, lr):
        b = b - lr*dl_db
        return b
    
    def implement_linear_regression(self, x_data, y_data, epoch_max = 50, lr = 1e-5, loss_type='mse'):
        losses = []
        
        w1, w2, w3, b = self.initialize_params()
        
        N = len(y_data)
        for _ in range(epoch_max):
            for i in range(N):
                # get a sample
                x1 = x_data[0][i]
                x2 = x_data[1][i]
                x3 = x_data[2][i]
                
                y = y_data[i]
                
                # compute output
                y_hat = self.predict(x1, x2, x3, w1, w2, w3, b)
                
                # compute loss
                loss = self.compute_loss(y=y, y_hat=y_hat, loss_type=loss_type)
                
                # compute gradient w1, w2, w3, b
                dl_dw1 = self.compute_gradient_wi(x1, y, y_hat)
                dl_dw2 = self.compute_gradient_wi(x2, y, y_hat)
                dl_dw3 = self.compute_gradient_wi(x3, y, y_hat)
                dl_db = self.compute_gradient_b(y, y_hat)
                
                # update parameters
                w1 = self.update_weight_wi(w1, dl_dw1, lr)
                w2 = self.update_weight_wi(w2, dl_dw2, lr)
                w3 = self.update_weight_wi(w3, dl_dw3, lr)
                b = self.update_weight_b(b, dl_db, lr)
                
                # logging
                losses.append(loss)
                
        return (w1, w2, w3, b, losses)
    
    def plot_result(self, losses, values=-1):
        plt.plot(losses[:values])
        plt.xlabel("#iteration")
        plt.ylabel("Loss")
        plt.show()
        
class NSampleLR(OneSampleLR):
    def implement_linear_regression_nsamples(self, x_data, y_data, epoch_max = 50, lr = 1e-5, loss_type='mse'):
        losses = []
        w1, w2, w3, b = self.initialize_params()
        
        N = len(y_data)
        for _ in range(epoch_max):
            
            loss_total = 0.0
            dw1_total = 0.0
            dw2_total = 0.0
            dw3_total = 0.0
            db_total = 0.0
            
            for i in range(N):
                # get a sample
                x1 = x_data[0][i]
                x2 = x_data[1][i]
                x3 = x_data[2][i]
                
                y = y_data[i]
                
                # compute output
                y_hat = self.predict(x1, x2, x3, w1, w2, w3, b)
                
                # compute loss
                loss = self.compute_loss(y=y, y_hat=y_hat, loss_type=loss_type)
                
                # accumulate loss
                loss_total = loss_total + loss
                
                # compute gradient w1, w2, w3, b
                dl_dw1 = self.compute_gradient_wi(x1, y, y_hat)
                dl_dw2 = self.compute_gradient_wi(x2, y, y_hat)
                dl_dw3 = self.compute_gradient_wi(x3, y, y_hat)
                dl_db = self.compute_gradient_b(y, y_hat)
                
                # accumulate gradient w1, w2, w3, b
                dw1_total = dw1_total + dl_dw1
                dw2_total = dw2_total + dl_dw2
                dw3_total = dw3_total + dl_dw3
                db_total = db_total + dl_db
                
            # update parameters
            w1 = self.update_weight_wi(w1, dw1_total/N, lr)
            w2 = self.update_weight_wi(w2, dw2_total/N, lr)
            w3 = self.update_weight_wi(w3, dw3_total/N, lr)
            b = self.update_weight_b(b, db_total/N, lr)
                
            # logging
            losses.append(loss_total/N)
                
        return (w1, w2, w3, b, losses)
    
if __name__ == '__main__':
    ## Testcases
    # One Sample LR
    # y_hat/y_pred
    regression = OneSampleLR()
    y = regression.predict(x1=1, x2=1, x3=1, w1=0, w2=0.5, w3=0, b=0.5)
    print(y) # 1.0
    
    # loss
    l = regression.compute_loss(y_hat=1, y=0.5, loss_type='mse')
    print(l) # 0.25
    
    # Gradient
    g_wi = regression.compute_gradient_wi(xi=1.0, y=1.0, y_hat=0.5)
    print(g_wi) # -1.0
    
    g_b = regression.compute_gradient_b(y=2.0, y_hat=0.5)
    print(g_b) # -3.0
    
    # Update parameters
    after_wi = regression.update_weight_wi(wi=1.0, dl_dwi=-0.5, lr=1e-5)
    print(after_wi) # 1.000005
    
    after_b = regression.update_weight_b(b=0.5, dl_db=-1.0, lr=1e-5)
    print(after_b) # 0.50001
    
    # plot 100 loss values
    X, y = prepare_data('./Module_4/Week_1/Data/advertising.csv') # Ensure this path is correct and points to your advertising.csv file
    (w1, w2, w3, b, losses) = regression.implement_linear_regression(X, y)
    regression.plot_result(losses=losses, values=100)
    
    # w1, w2, w3
    print(w1, w2, w3) # 0.07405984066396477 0.15917360263437663 0.017561197559948935
    
    # given new data 
    tv = 19.2
    radio = 35.9
    newspaper = 51.3
    
    sales = regression.predict(tv, radio, newspaper, w1, w2, w3, b)
    print(f'Predicted sales is {sales}')
    
    # plot 100 loss values with MAE loss
    l_mae = regression.compute_loss(y_hat=1, y=0.5, loss_type='mae')
    print(l_mae) # 1.0
    
    (w1, w2, w3, b, losses) = regression.implement_linear_regression(X, y, loss_type='mae')
    regression.plot_result(losses=losses, values=100)
    
    # N sample LR
    regression = NSampleLR()
    (w1, w2, w3, b, losses) = regression.implement_linear_regression_nsamples(X, y, loss_type='mse', epoch_max=1000, lr=1e-5)
    print(f"w1, w2, w3 = {w1, w2, w3}")
    regression.plot_result(losses=losses)
    
    (w1, w2, w3, b, losses) = regression.implement_linear_regression_nsamples(X, y, loss_type='mae', epoch_max=1000, lr=1e-5)
    regression.plot_result(losses=losses)
    