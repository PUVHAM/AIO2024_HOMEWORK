import numpy as np
import pandas as pd

#########################
# Create data
#########################
def create_train_data():
    df = pd.read_csv("./Module_2/Week_3/Data/PlayTennis.csv") # Ensure this path is correct and points to your PlayTennis.csv file
    data = df.to_numpy()
    return np.array(data)

# This function below is used to calculate prior probability P(play tennis = 'No') and P(play tennis = 'Yes')
def compute_prior_probability(train_data):
    y_unique = ['no','yes']
    prior_probability = np.zeros(len(y_unique))
    for num in range(0, len(y_unique)):
        prior_probability[num] = np.sum(train_data[:,-1] == y_unique[num]) / len(train_data)
    return prior_probability

# This function calculates the conditional probability P(feature|class) for each feature in the dataset
def compute_conditional_probability(train_data):
    y_unique = ['no','yes']
    conditional_probability = []
    list_x_name = []
    for i in range(0, train_data.shape[1] - 1):
        x_unique = np.unique(train_data[:,i])
        list_x_name.append(x_unique)

        x_conditional_probability = np.zeros((len(y_unique),len(x_unique)))
        
        for rows in range(0, len(y_unique)):
            for cols in range(0, len(x_unique)):
                x_conditional_probability[rows, cols] = np.sum((train_data[:,-1] == y_unique[rows]) & (train_data[:,i] == x_unique[cols])) / np.sum(train_data[:,-1] == y_unique[rows])
        
        conditional_probability.append(x_conditional_probability)
    
    return conditional_probability, list_x_name

# This function is used to return the index of the feature name
def get_index_from_value(feature_name, list_features) :
    return np.nonzero(list_features == feature_name)[0][0]

# Test function create_train_data()
train_data = create_train_data()
print(f"Data: \n{train_data}")

# Test function compute_prior_probability()
prior_probablity = compute_prior_probability(train_data)
print("P(play tennis = 'No') =", prior_probablity[0])
print("P(play tennis = 'Yes') =", prior_probablity[1])

# Test function compute_conditional_probability()
_, list_x_name = compute_conditional_probability(train_data)
print("x1 = ", list_x_name[0])
print("x2 = ", list_x_name[1])
print("x3 = ", list_x_name[2])
print("x4 = ", list_x_name[3])

# Test function get_index_from_value()
outlook = list_x_name[0]
i1 = get_index_from_value("Overcast", outlook)
i2 = get_index_from_value("Rain", outlook)
i3 = get_index_from_value("Sunny", outlook)

print(i1, i2, i3)

# Testcases
train_data = create_train_data()
conditional_probability, list_x_name = compute_conditional_probability(train_data=train_data)
x1 = get_index_from_value("Sunny", list_x_name[0])

## Compute P("Outlook" = "Sunny"|"Play Tennis"= "Yes")
print(f"P('Outlook' = 'Sunny'|'Play Tennis'= 'Yes') = {round(conditional_probability[0][1,x1],2)}")

## Compute P("Outlook" = "Sunny"|"Play Tennis"= "No")
print(f"P('Outlook' = 'Sunny'|'Play Tennis'= 'No') = {round(conditional_probability[0][0,x1],2)}")

###########################
# Train Naive Bayes Model
###########################
def train_naive_bayes(train_data):
    # Step 1: Caculate Prior Probability
    prior_probablity = compute_prior_probability(train_data=train_data)
    
    # Step 2: Calculate Conditional Probability
    conditional_probability, list_x_name = compute_conditional_probability(train_data=train_data)
    
    return prior_probablity, conditional_probability, list_x_name

####################
# Prediction
####################
def prediction_play_tennis(x, list_x_name, prior_probability, conditional_probability):
    
    x1 = get_index_from_value(x[0], list_x_name[0])
    x2 = get_index_from_value(x[1], list_x_name[1])
    x3 = get_index_from_value(x[2], list_x_name[2])
    x4 = get_index_from_value(x[3], list_x_name[3])
    
    p0 = prior_probability[0]\
    *conditional_probability[0][0,x1]\
    *conditional_probability[1][0,x2]\
    *conditional_probability[2][0,x3]\
    *conditional_probability[3][0,x4]

    p1 = prior_probability[1]\
    *conditional_probability[0][1,x1]\
    *conditional_probability[1][1,x2]\
    *conditional_probability[2][1,x3]\
    *conditional_probability[3][1,x4]
    
    return 0 if p0 > p1 else 1

# Testcases
x = ['Sunny', 'Cool', 'High', 'Strong']
data = create_train_data()
prior_probablity, conditional_probability, list_x_name = train_naive_bayes(data)
pred = prediction_play_tennis(x, list_x_name, prior_probablity, conditional_probability)

if(pred):
    print("Ad should go!")
else:
    print("Ad should not go!")




        
        

