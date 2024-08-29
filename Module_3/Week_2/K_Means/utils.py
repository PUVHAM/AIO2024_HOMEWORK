from sklearn.datasets import load_iris

def load_iris_data():
    iris_dataset = load_iris()
    data = iris_dataset.data[:, :2]  # Selecting the first two features
    return data
