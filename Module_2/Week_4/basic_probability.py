import numpy as np

# Question 1: The output of the following program to calculate mean:
# Given Data X = {2, 0, 2, 2, 7, 4, −2, 5, −1, −1}
# Complete the function compute_mean() to calculate the mean μ of the given X.
def compute_mean(x):
    mean = np.mean(x)
    return mean

# Question 2: The result of the following program to calculate median:
# Given Data X = {1, 5, 4, 4, 9, 13}. 
# Complete the function compute_median() to find the median of the given X.
def compute_median(x):
    size = len(x)
    x = np.sort(x)
    print(x)
    if size % 2 == 0:
        return (x[int(size / 2 - 1)] + x[int(size / 2)]) / 2
    else:
        return x[int((size + 1) / 2 - 1)]

# Question 3: The result of the following program to calculate variance and standard deviation:
# Given Data X = {171, 176, 155, 167, 169, 182}
# Complete the function compute_std() to find the standard deviation σ of the given X.
def compute_std(x):
    mean = compute_mean(x)
    variance = np.mean((x - mean) ** 2)
    return np.sqrt(variance)

# Question 4: The result of the following program to calculate correlation coefficient:
# Given Data X = {−2, −5, −11, 6, 4, 15, 9} and
# Y = { 4, 25, 121, 36, 16, 225, 81}
# Complete the function compute_correlation_cofficient() to find the correlation coefficient of the given X and Y?
def compute_correlation_cofficient(x, y):
    N = len(x)
    numerator = N * np.sum(x*y) - np.sum(x) * np.sum(y)
    denominator = np.sqrt((N * np.sum(np.power(x, 2)) - (np.sum(x))**2) * (N * np.sum(np.power(y, 2)) - (np.sum(y)) ** 2))
    return np.round(numerator / denominator, 2)


if __name__ == "__main__":
    # Question 1
    x = [2, 0, 2, 2, 7, 4, -2, 5, -1, -1]
    print("Mean:", compute_mean(x)) # Mean: 1.8
    
    # Question 2
    x = [1, 5, 4, 4, 9, 13]
    print(f"Median: {compute_median(x)}") # Median: 4.5
    
    # Question 3
    x = [171, 176, 155, 167, 169, 182]
    print(f"Standard deviation: {round(compute_std(x), 2)}") # Standard deviation: 8.33
    
    # Question 4
    x = np.asarray([-2, -5, -11, 6, 4, 15, 9])
    y = np.asarray([4, 25, 121, 36, 16, 225, 81])
    print("Correlation:", compute_correlation_cofficient(x, y)) # Correlation: 0.42