import random
import math

# Function to print loss values for a given number of samples and calculate the overall loss
def print_loss_with_nums_sample(num, loss_name):
    random.seed(0)
    loss = []
    
    for i in range(num):
        yp = random.uniform(0, 10)
        yt = random.uniform(0, 10)
        abs_error = abs(yp - yt)
        square_error = (yp - yt)**2
        
        # Print the loss details
        if loss_name == 'MAE':
            print(f"loss name:  {loss_name}, sample : {i + 1}, pred : {yp}, target : {yt}, loss: {abs_error}")
            loss.append(abs_error)
        elif loss_name == 'MSE' or loss_name == 'RMSE':
            print(f"loss name:  {loss_name}, sample : {i + 1}, pred : {yp}, target : {yt}, loss: {square_error}")
            loss.append(square_error)
    
    # Calculate the overall loss
    result = 0
    if loss_name == 'MAE':
        result = (1/num) * sum(loss)
    elif loss_name == 'MSE':
        result = (1/num) * sum(loss)
    elif loss_name == 'RMSE':
        result = math.sqrt((1/num) * sum(loss))
        
    return f"final {loss_name}: {result}"

# Function to get user input and call the print_loss_with_nums_sample function
def exercise3():
    x = input("Input number of samples (integer number) which are generated: ")
    if not x.isnumeric():
        return "number of samples must be an integer number"

    x = int(x)
    loss_function = input("Input loss name: ")
    if loss_function not in ['MAE', 'MSE', 'RMSE']:
        return "loss name must be one of 'MAE', 'MSE', or 'RMSE'"
    
    result = print_loss_with_nums_sample(x, loss_function)
    return result
    
print(exercise3())
