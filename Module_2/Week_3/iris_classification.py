import numpy as np

#########################
# Create data
#########################
def create_train_data_iris():
    data = np.loadtxt('./Module_2/Week_3/Data/iris.data.txt', delimiter=',', dtype=str) # Ensure this path is correct and points to your iris.data.txt file
    return data

# This function below is used to calculate prior probability 
def compute_prior_probability_iris(train_data):
    y_unique = np.unique(train_data[:, -1])
    prior_probability = np.zeros(len(y_unique))
    for num in range(0, len(y_unique)):
        prior_probability[num] = np.sum(train_data[:, -1] == y_unique[num]) / len(train_data)
    return prior_probability

# This function is used to compute the conditional probabilities
def compute_conditional_probability_iris(train_data):
    y_unique = np.unique(train_data[:, -1])
    x_feature = len(train_data[1, :]) - 1
    conditional_probability = []
    for rows in range(0, x_feature):
        x_conditional_probability = np.zeros((len(y_unique), 2))
        for cols in range(0, len(y_unique)):
            mean = np.mean(train_data[:,rows][train_data[:,-1] == y_unique[cols]].astype(float)) 
            sigma = np.std(train_data[:,rows][train_data[:,-1] == y_unique[cols]].astype(float)) 
            variance = sigma ** 2
            x_conditional_probability[cols] = [mean, variance]
    
        conditional_probability.append(x_conditional_probability)
    return conditional_probability

# Define the Gaussian function
def guassian_distribution(x, mean, variance):
    result = (1.0 / np.sqrt(2*np.pi*variance))\
            * np.exp(-((float(x) - mean) ** 2) / (2 * variance))
    return result

###########################
# Train Naive Bayes Model
###########################
def train_gaussian_naive_bayes(train_data):
    # Step 1: Calculate Prior Probability
    prior_probability = compute_prior_probability_iris(train_data)

    # Step 2: Calculate Conditional Probability
    conditional_probability  = compute_conditional_probability_iris(train_data)

    return prior_probability, conditional_probability

####################
# Prediction
####################
def prediction_iris(x, prior_probability, conditional_probability):
    num_classes = len(prior_probability)
    num_features = len(x)

    list_p = []
    for class_idx in range(num_classes):
        p = prior_probability[class_idx]
        for feature_idx in range(num_features):
            mean = conditional_probability[feature_idx][class_idx][0]
            variance = conditional_probability[feature_idx][class_idx][1]
            p *= guassian_distribution(x[feature_idx], mean, variance)
            
        list_p.append(p)
    
    return list_p.index(max(list_p))

# Testcases
x = [6.3 , 3.3, 6.0,  2.5]
train_data = create_train_data_iris()
y_unique = np.unique(train_data[:,4])
prior_probability, conditional_probability = train_gaussian_naive_bayes(train_data)
pred = y_unique[prediction_iris(x, prior_probability, conditional_probability)]
assert pred == "Iris-virginica"

x = [5.0,2.0,3.5,1.0]
train_data = create_train_data_iris()
y_unique = np.unique(train_data[:,4])
prior_probability, conditional_probability = train_gaussian_naive_bayes(train_data)
pred = y_unique[prediction_iris(x, prior_probability, conditional_probability)]
assert pred == "Iris-versicolor"

x = [4.9,3.1,1.5,0.1]
train_data = create_train_data_iris()
y_unique = np.unique(train_data[:,4])
prior_probability, conditional_probability = train_gaussian_naive_bayes(train_data)
pred = y_unique[prediction_iris(x, prior_probability, conditional_probability)]
assert pred == "Iris-setosa"