from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# Load dataset
machine_cpu = fetch_openml(name='machine_cpu')
machine_data = machine_cpu.data
machine_labels = machine_cpu.target 

# Split train:test = 8:2
x_train, x_test, y_train, y_test = train_test_split(
    machine_data,
    machine_labels,
    test_size=0.2,
    random_state=42
)

# Define model
tree_reg = DecisionTreeRegressor(random_state=42, criterion='squared_error', ccp_alpha=0.01)

# Train
tree_reg.fit(x_train, y_train)

# Predict and evaluate
y_pred = tree_reg.predict(x_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}") # Mean Squared Error: 9293.464285714286

