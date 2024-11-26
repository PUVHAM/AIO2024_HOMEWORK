import numpy as np
from gradient_descent import train_pl

def rmsprop(w, dw, lr, s, gamma):
    """Update w1, w2 using RMSProp
    
    Args:
    w (numpy.ndarray): [w1, w2]
    dw (numpy.ndarray): [dw1, dw2]
    lr (float): learning rate
    s (numpy.ndarray): [v1, v2] Exponentially weighted averages squared gradients
    gamma (float): long-range average
    Returns:
    w (numpy.ndarray): [w1, w2] updated
    s (numpy.ndarray): [v1, v2] Exponentially weighted averages squared updated
    """
    epsilon = 1e-6
    s = gamma * s + (1 - gamma) * (dw ** 2)
    w = w - lr * dw / (np.sqrt(s + epsilon))
    return w, s

if __name__ == "__main__":
    print(f"w1, w2 after 30 epochs: {train_pl('RMSProp', 0.3, 30)[-1]}") # w1, w2 after 30 epochs: [-3.0057612e-03 -3.0051769e-17]