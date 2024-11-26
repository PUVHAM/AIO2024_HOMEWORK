import numpy as np
from gradient_descent import train_pl

def adam(w, dw, lr, v, s, beta_1, beta_2, t):
    """Update w1, w2 using Gradient Descent + Momentum
    
    Args:
    w (numpy.ndarray): [w1, w2]
    dw (numpy.ndarray): [dw1, dw2]
    lr (float): learning rate
    v (numpy.ndarray): [v1, v2] Exponentially weighted averages gradients
    s (numpy.ndarray): [v1, v2] Exponentially weighted averages squared gradients
    beta_1 (float): long-range average for v
    beta_2 (float): long-range average for s
    Returns:
    w (numpy.ndarray): [w1, w2] updated
    v (numpy.ndarray): [v1, v2] Exponentially weighted averages gradients updated
    s (numpy.ndarray): [v1, v2] Exponentially weighted averages squared updated
    """
    epsilon = 1e-6
    v = beta_1 * v + (1 - beta_1) * dw 
    s = beta_2 * s + (1 - beta_2) * (dw ** 2)
    v_corr = v / (1 - beta_1 ** t)
    s_corr = s / (1 - beta_2 ** t)
    w = w - lr * v_corr / (np.sqrt(s_corr) + epsilon)
    return w, v, s

if __name__ == "__main__":
    print(f"w1, w2 after 30 epochs: {train_pl('Adam', 0.2, 30)[-1]}") # w1, w2 after 30 epochs: [-0.11386538  0.06793538]