from utils import load_iris_data
import matplotlib.pyplot as plt

# Load data
data = load_iris_data()

# Plot data
plt.scatter(data[:, 0], data[:, 1], c='gray')
plt.title("Initial Dataset")
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()