import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# Load the diabetes dataset
diabetes_x, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Split train:test = 8:2
x_train, x_test, y_train, y_test = train_test_split(
    diabetes_x,
    diabetes_y,
    test_size=0.2,
    random_state=42
)

# Scale the features using StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Build KNN model
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(x_train, y_train)

# Predict and evaluate test set
y_pred = knn_regressor.predict(x_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}") # Mean Squared Error: 3047.449887640449
print(f"Root Mean Squared Error: {rmse}") # Root Mean Squared Error: 55.203712625515045
print(f"R2 Score: {r2}") # R2 Score: 0.42480887066066253