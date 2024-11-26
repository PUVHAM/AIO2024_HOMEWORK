import numpy as np
from gradient_descent import train_pl

def sgd_momentum(w, dw, lr, v, beta):
    """Update w1, w2 using Gradient Descent + Momentum
    
    Args:
    w (numpy.ndarray): [w1, w2]
    dw (numpy.ndarray): [dw1, dw2]
    lr (float): learning rate
    v (numpy.ndarray): [v1, v2] Exponentially weighted averages gradients
    beta (float): long-range average
    Returns:
    w (numpy.ndarray): [w1, w2] updated
    v (numpy.ndarray): [v1, v2] Exponentially weighted averages gradients updated
    """
    v = beta * v + (1 - beta) * dw
    w = w - lr * v
    return w, v

if __name__ == "__main__":
    print(f"w1, w2 after 30 epochs: {train_pl('Gradient Descent With Momentum', 0.6, 30)[-1]}") # w1, w2 after 30 epochs: [-6.100721e-02  6.451628e-05]