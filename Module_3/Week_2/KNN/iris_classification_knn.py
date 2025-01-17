from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load the iris dataset
iris_x, iris_y = datasets.load_iris(return_X_y=True)

# Split train:test = 8:2
x_train, x_test, y_train, y_test = train_test_split(
    iris_x,
    iris_y,
    test_size=0.2,
    random_state=42
)

# Scale the features using StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Build KNN classfier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(x_train, y_train)

# Predict and evaluate test set
y_pred = knn_classifier.predict(x_test)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}") # Accuracy score: 1.0