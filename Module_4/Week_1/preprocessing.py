import numpy as np

def get_column(data, index):
    result = []
    for vector in data:
        result.append(vector[index])
    return result

def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',', skip_header=1).tolist()
    
    # get tv (idx = 0)
    tv_data = get_column(data, 0)
    
    # get radio (idx = 1)
    radio_data = get_column(data, 1)
    
    # get newspaper (idx = 2)
    newspaper_data = get_column(data, 2)
    
    # get sales (idx = 3)
    sales_data = get_column(data, 3)
    
    # building X input and y output for training
    X = [tv_data, radio_data, newspaper_data]
    y = sales_data
    return X, y

def advanced_prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',', skip_header=1).tolist()
    
    # get tv (idx = 0)
    tv_data = get_column(data, 0)
    
    # get radio (idx = 1)
    radio_data = get_column(data, 1)
    
    # get newspaper (idx = 2)
    newspaper_data = get_column(data, 2)
    
    # get sales (idx = 3)
    sales_data = get_column(data, 3)
    
    # building X input and y output for training
    # Create list of features for input
    X = [[1, x1, x2, x3] for x1, x2, x3 in zip(tv_data, radio_data, newspaper_data)]
    y = sales_data
    return X, y
if __name__ == '__main__':
    # Testcases
    X, y = prepare_data('./Module_4/Week_1/Data/advertising.csv') # Ensure this path is correct and points to your advertising.csv file
    lst = [sum(X[0][:5]), sum(X[1][:5]), sum(X[2][:5]), sum(y[:5])]
    print(lst) # [624.1, 175.10000000000002, 300.5, 78.9]
    