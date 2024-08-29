import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

# Load IMDB dataset
imdb = load_dataset("imdb")
imdb_train, imdb_test = imdb["train"], imdb["test"]

# Convert text to vector using BOW
vectorizer = CountVectorizer(max_features=1000)
x_train = vectorizer.fit_transform(imdb_train['text']).toarray()
x_test = vectorizer.transform(imdb_test['text']).toarray()
y_train = np.array(imdb_train['label'])
y_test = np.array(imdb_test['label'])

# Scale the features using StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Build KNN classfier
knn_classifier = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree')
knn_classifier.fit(x_train, y_train)

# Predict and evaluate test set
y_pred = knn_classifier.predict(x_test)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}") # Accuracy score: 0.60444