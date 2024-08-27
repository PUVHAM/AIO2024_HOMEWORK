import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#########################
# Load data
#########################
dataset_path = './Module_3/Week_1/Data/IMDB-Movie-Data.csv' # Ensure this path is correct and points to your IMDB-Movie-Data.csv file

## Read data from .csv file
data = pd.read_csv(dataset_path)

# Preview top 5 rows using head()
print("Top 5:\n", data.head())

# Let's first understand the basic information about this data
print("Data information:")
print(data.info())
print(data.describe(), "\n")


#########################
# Data Selection – Indexing and Slicing data
#########################

# Extract data as series
genre = data['Genre']
print("Data as Series:\n", genre, "\n")

# Extract data as dataframe
print("Data as DataFrame:\n", data[['Genre']])

# Select and split multiple columns at once, thus creating a new DataFrame
some_cols = data[['Title', 'Genre', 'Actors', 'Director', 'Rating']]
print(f"New DataFrame:\n {some_cols}")

# Slicing from row 10 to row 15
print(f"Slicing:\n {data.iloc[10:15][['Title', 'Rating', 'Revenue (Millions)']]}")


#########################
# Data Selection – Based on Conditional filtering
#########################

# Take movies from 2010 to 2015, with ratings less than 6.0 but revenue in the top 5% of the entire dataset.
revenue_column = data['Revenue (Millions)']
print("Select based on condition:\n")
print(data[((data['Year'] >= 2010) & (data['Year'] <= 2015)) 
        & (data['Rating'] < 6.0)
        & (revenue_column > revenue_column.quantile(0.95))])


#########################
# Groupby Operations
#########################

# Find the average rating achieved by directors by grouping the Ratings of movies by Director.
print(f"Groupby:\n {data.groupby('Director')[['Rating']].mean().head()}")


#########################
# Sorting Operations
#########################

# Find top 5 directors have the most average rating based on groupby result
print("\nSorting:")
print(data.groupby('Director')[['Rating']].mean().sort_values(['Rating'], ascending=False).head())


#########################
# View missing values
#########################

# To check null values row-wise
print(f"\nCheck NULL:\n{data.isnull().sum()}")


#########################
# Deal with missing values - Deleting
#########################

# Use drop function to drop columns
drop_columns = data.drop('Metascore', axis =1)
print(f"\nNew DataFrame: \n{drop_columns.head()}")

# Drop null rows
drop_rows = data.dropna()
print(f"\nNew DataFrame: \n{drop_rows.head()}")

#########################
# Deal with missing values - Filling
#########################

# Some rows in the Revenue column have null values, thus assigning them average value
revenue_mean = revenue_column.mean()
print("The mean revenue is: ", revenue_mean)

print("\nFilling NULL:")
revenue_column.fillna(revenue_mean, inplace=True)
print(data.isnull().sum()[['Revenue (Millions)']])


#########################
# apply() functions
#########################

# Classify movies based on ratings
def rating_group(rating):
    if rating >= 7.5:
        return 'Good'
    elif rating >= 6.0:
        return 'Average'
    else:
        return 'Bad'
    
# Lets apply this function on our movies data
# Creating a new variable in the dataset to hold the rating category
data['Rating_category'] = data['Rating'].apply(rating_group)

print("\nDataFrame after adding Rating_category column:")
print(data[['Title', 'Director', 'Rating', 'Rating_category']].head())