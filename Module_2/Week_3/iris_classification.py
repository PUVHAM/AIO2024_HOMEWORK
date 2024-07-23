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
            sigma = sigma ** 2
            x_conditional_probability[cols] = [mean, sigma]
    
        conditional_probability.append(x_conditional_probability)
    return conditional_probability

# Define the Gaussian function
def guassian_distribution(x, mean, sigma_squared):
    result = (1.0 / np.sqrt(2*np.pi*sigma_squared))\
            * np.exp((-1/2)*np.power((float(x)-mean)/sigma_squared,2))
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
def prediction_iris(x,  prior_probability, conditional_probability):
    p0 = prior_probability[0]\
    *guassian_distribution(x[0], conditional_probability[0][0][0],conditional_probability[0][0][1])\
    *guassian_distribution(x[1], conditional_probability[1][0][0],conditional_probability[1][0][1])\
    *guassian_distribution(x[2], conditional_probability[2][0][0],conditional_probability[2][0][1])\
    *guassian_distribution(x[3], conditional_probability[3][0][0],conditional_probability[3][0][1])

    p1 = prior_probability[1]\
    *guassian_distribution(x[0], conditional_probability[0][1][0],conditional_probability[0][1][1])\
    *guassian_distribution(x[1], conditional_probability[1][1][0],conditional_probability[1][1][1])\
    *guassian_distribution(x[2], conditional_probability[2][1][0],conditional_probability[2][1][1])\
    *guassian_distribution(x[3], conditional_probability[3][1][0],conditional_probability[3][1][1])

    p2 = prior_probability[2]\
    *guassian_distribution(x[0], conditional_probability[0][2][0],conditional_probability[0][2][1])\
    *guassian_distribution(x[1], conditional_probability[1][2][0],conditional_probability[1][2][1])\
    *guassian_distribution(x[2], conditional_probability[2][2][0],conditional_probability[2][2][1])\
    *guassian_distribution(x[3], conditional_probability[3][2][0],conditional_probability[3][2][1])

    list_p = [p0, p1, p2]

    return list_p.index(np.max(list_p))

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