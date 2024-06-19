import math

def is_number(n):
    try:
        float(n)  # Type-casting the string to ‘float‘
        # If string is not a valid ‘float‘, it’ll raise ‘ValueError‘ exception1
    except ValueError:
        return False
    return True

def exercise2():
    # Input x and activate_function
    x = input("Input x = ")
    if not is_number(x):
        return "Invalid input: x must be a number."
    
    x = float(x)
    activation_function = input("Input activation Function (sigmoid|relu|elu): ")
    
    alpha = 0.01
    
    #Check and calculate activation_function 
    if activation_function == 'sigmoid':
        result = 1 / (1 + math.exp(-x))
    elif activation_function == 'relu':
        if x <= 0:
            result = 0
        else:
            result = x
    elif activation_function == 'elu':
        if x <= 0:
            result = alpha * (math.exp(x) - 1)
        else:
            result = x
    else: 
         return f'{activation_function} is not supported'
     
    return result

print(exercise2())
    
    
    
    