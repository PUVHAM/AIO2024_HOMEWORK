import numpy as np

# we have function f(w1, w2) = 0.1*(w1^2) + 2*(w2^2) (1)
def df_w(w):
    """Compute gradient dw1, dw2
    
    Args:
        w (numpy.ndarray): [w1, w2]
    Returns:
        dw (numpy.ndarray): [dw1, dw2]
    """
    dw_1 = 2*0.1*w[0]
    dw_2 = 2*2*w[1]
    dw = np.array([dw_1, dw_2])
    return dw

def sgd(w, dw, lr):
    """Update w1, w2 using Gradient Descent
    
    Args:
        w (numpy.ndarray): [w1, w2]
        dw (numpy.ndarray): [dw1, dw2]
        lr (float): learning rate
    Returns:
        w (numpy.ndarray): [w1, w2] updated
    """
    return w - lr*dw

def train_pl(optimizer, lr, epochs):
    """Find minimum of function (1) based on optimization algorithms

    Args:
        optimizer: optimization function
        lr (float): learning rate
        epochs (int): number of repetitions
    Returns:
        results (list): [w1, w2] after each epoch
    """
    from gradient_descent_momentum import sgd_momentum
    from rmsprop_algorithm import rmsprop
    from adam_algorithm import adam
    
    # Initial point
    w = np.array([-5, -2], dtype=np.float32)
    v = np.array([0, 0], dtype=np.float32)
    s = np.array([0, 0], dtype=np.float32)
    # List of results
    results = [w]
    
    for epoch in range(epochs):
        dw = df_w(w)
        if optimizer == "Gradient Descent":
            w = sgd(w, dw, lr)
        elif optimizer == "Gradient Descent With Momentum":
            w, v = sgd_momentum(w, dw, lr, v, 0.5)
        elif optimizer == "RMSProp":
            w, s = rmsprop(w, dw, lr, s, 0.9)
        elif optimizer == "Adam":
            w, v, s = adam(w, dw, lr, v, s, 0.9, 0.999, epoch + 1)
        results.append(w)
    return results

if __name__ == "__main__":
    print(f"w1, w2 after 30 epochs: {train_pl('Gradient Descent', 0.4, 30)[-1]}") # w1, w2 after 30 epochs: [-4.0983096e-01 -4.4214812e-07]
            