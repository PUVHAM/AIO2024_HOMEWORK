import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame and convert it to a NumPy array
df = pd.read_csv('./Module_2/Week_1/Data/advertising.csv')
data = df.to_numpy()

# Get the maximum value and its corresponding index in the Sales column:
print(f"Max: {np.max(data[:,3])} - Index: {np.argmax(data[:,3])}\n")

# Mean value of the TV column:
print(np.mean(data[:,0]))

# Count of records with values in the Sales column greater than or equal to 20:
# Method 1
count = sum([1 for value in data[:,3] if value >= 20])
print(count)
# Method 2
count = np.sum(data[:,3] >= 20)
print(count)

# Calculate the mean value of the Radio column where the corresponding Sales values are greater than or equal to 15:
mean_value = np.mean(data[:,1], where=data[:,3] >= 15)
print(mean_value)

# Sum of the Sales column with the condition that the Newspaper values are greater than the mean of the Newspaper column:
sum_value = np.sum(data[:, 3], where=(data[:, 2] > np.mean(data[:, 2])))
print(sum_value)

# Let A be the mean value of the Sales column. Create a new array scores containing 'Good', 'Average', and 'Bad' values based on the conditions:
# If the current value > A => 'Good'
# If the current value < A => 'Bad'
# If the current value == A => 'Average'
# Then print the results for scores[7:10]
a = np.mean(data[:, 3])
def classify_scores(a, start, end):
    scores = []
    for value in data[:, 3]:
        if value > a:
            scores.append('Good')
        elif value < a:
            scores.append('Bad')
        else:
            scores.append('Average')
    return scores[start:end]
print(classify_scores(a, 7, 10))

# Let A be the value on the Sales column closest to the mean value of the Sales column. Create a new array scores containing 'Good', 'Average', and 'Bad' values based on the conditions:
# If the current value > A => 'Good'
# If the current value < A => 'Bad'
# If the current value == A => 'Average'
# Then print the results for scores[7:10]
a = data[:, 3][np.abs(data[:, 3] - np.mean(data[:, 3])).argmin()]
print(classify_scores(a, 7, 10))
