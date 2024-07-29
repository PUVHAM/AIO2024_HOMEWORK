import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import cv2
from basic_probability import compute_correlation_cofficient as correlation

data = pd.read_csv('./Module_2/Week_1/Data/advertising.csv')

# Question 5: Result of the following code:
x = data['TV']
y = data['Radio']
corr_xy = correlation(x, y)
print(f"Correlation between TV and Sales: {round(corr_xy, 2)}") # Correlation between TV and Sales: 0.05

# Question 6: Result of the following code:
features = ['TV', 'Radio', 'Newspaper']
for feature_1 in features:
    for feature_2 in features:
        correlation_value = correlation(data[feature_1], data[feature_2])
        print(f"Correlation between {feature_1} and {feature_2}: {round(correlation_value, 2)}")    # Output: Correlation between TV and TV: 1.0
                                                                                                    #         Correlation between TV and Radio: 0.05
                                                                                                    #         Correlation between TV and Newspaper: 0.06      
                                                                                                    #         Correlation between Radio and TV: 0.05
                                                                                                    #         Correlation between Radio and Radio: 1.0        
                                                                                                    #         Correlation between Radio and Newspaper: 0.35   
                                                                                                    #         Correlation between Newspaper and TV: 0.06      
                                                                                                    #         Correlation between Newspaper and Radio: 0.35   
                                                                                                    #         Correlation between Newspaper and Newspaper: 1.0
        

# Question 7: Provide the code that matches the following result:
# data = pd.read_csv("advertising.csv")
# x = data['Radio']
# y = data['Newspaper']
#
# result = # Your code here #
# print(result)
#
# Expected output: [[1.         0.35410375]
#                   [0.35410375 1.        ]]

x = data['Radio']
y = data['Newspaper']

result = np.corrcoef(x, y) # The correct code that matches the result is np.corrcoef(x, y)
print(result)

# Question 8: Provide the code that matches the following result
#                  TV     Radio  Newspaper     Sales
# TV         1.000000  0.054809   0.056648  0.901208
# Radio      0.054809  1.000000   0.354104  0.349631
# Newspaper  0.056648  0.354104   1.000000  0.157960
# Sales      0.901208  0.349631   0.157960  1.000000
print(data.corr())

# Question 9: Provide the code that matches the following result:
# Using cv2 to display the expected output
expected_output = cv2.imread('./Module_2/Week_4/Data/Heatmap.png', 1)
expected_output= cv2.cvtColor(expected_output, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(expected_output)
plt.title('Expected Output')

# Code that matches the expected output 
data_corr = data.corr()
plt.figure(figsize=(5,4))
sns.heatmap(data_corr, annot=True, fmt=".2f", linewidth=.5)
plt.title('Generated Heatmap')
plt.show()