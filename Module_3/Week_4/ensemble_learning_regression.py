import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
dataset_path = './Module_3/Week_4/Data/Housing.csv' # Ensure this path is correct and points to your Housing.csv file
df = pd.read_csv(dataset_path)

# Pre-process categorical
categorical_cols = df.select_dtypes(include=['object']).columns.to_list()
# print(categorical_cols) // ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

ordinal_encoder = OrdinalEncoder()
encoded_categorical_cols = ordinal_encoder.fit_transform(
    df[categorical_cols]
)
encoded_categorical_df = pd.DataFrame(
    encoded_categorical_cols,
    columns=categorical_cols
)
numerical_df = df.drop(categorical_cols, axis=1)
encoded_df = pd.concat(
    [numerical_df, encoded_categorical_df], 
    axis=1
)

# Normalize dataset
normalizer = StandardScaler()
dataset_arr = normalizer.fit_transform(encoded_df)

# Split X, y
X, y = dataset_arr[:, 1:], dataset_arr[:, 0]

# Split train:val = 7:3
random_state = 1
x_train, x_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.3,
    random_state=random_state,
    shuffle=True
)

def train_and_evaluate_model(model, x_train, y_train, x_val, y_val):
    # Train the model
    model.fit(x_train, y_train)
    
    # Predict on validation set
    y_pred = model.predict(x_val)
    
    # Calculate MAE and MSE
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    
    # Print the results
    print(f'Model: {model.__class__.__name__}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print('--------------------------')

if __name__ == '__main__':
    # Random Forest algorithm
    regressor = RandomForestRegressor(random_state=random_state, min_samples_leaf=1, max_features=None)
    train_and_evaluate_model(regressor, x_train, y_train, x_val, y_val)

    # AdaBoost algorithm
    regressor = AdaBoostRegressor(random_state=random_state, 
                                learning_rate=0.1)
    train_and_evaluate_model(regressor, x_train, y_train, x_val, y_val)

    # Gradient Boosting algorithm
    regressor = GradientBoostingRegressor(random_state=random_state, 
                                learning_rate=0.2)
    train_and_evaluate_model(regressor, x_train, y_train, x_val, y_val)