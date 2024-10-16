import numpy as np 

def load_data_from_file(file_name = './Module_4/Week_2/Data/advertising.csv'):
    data = np.genfromtxt(file_name, dtype=None, delimiter=',', skip_header=1)
    features_x = data[:, :3]
    sales_y = data[:, 3]

    features_x = np.hstack((np.ones((features_x.shape[0], 1)), features_x))

    return features_x, sales_y

if __name__ == "__main__":
    features_x, sale_y = load_data_from_file()
    print(features_x[:5, :])
    print(sale_y.shape)    