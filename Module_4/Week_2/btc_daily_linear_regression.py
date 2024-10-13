import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # type: ignore
from mplfinance.original_flavor import candlestick_ohlc
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 

# Load dataset
df = pd.read_csv('./Module_4/Week_2/Data/BTC-Daily.csv') # Ensure this path is correct and points to your BTC-Daily.csv file

# Remove duplicate rows
df = df.drop_duplicates()

# Range of dates covered
df['date'] = pd.to_datetime(df['date'])
date_range = str(df['date'].dt.date.min()) + ' to ' + str(df['date'].dt.date.max())
print(date_range)
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

unique_years = df['date'].dt.year.unique()

for year in unique_years:
    year_month_day = pd.DataFrame({'date': pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')})
    year_month_day['year'] = year_month_day['date'].dt.year 
    year_month_day['month'] = year_month_day['date'].dt.month 
    year_month_day['day'] = year_month_day['date'].dt.day 
    
    merged_data = pd.merge(year_month_day, df, on=['year', 'month', 'day'], how='left', validate="many_to_many")
    if year == 2021: # This loop is used for this assignment only to plot year 2021, you can delete or change the loop
    # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(merged_data['date_x'], merged_data['close'])
        plt.title(f'Bitcoin Closing Prices - {year}')
        plt.xlabel('Date')
        plt.ylabel('Closing Price (USD)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
# Filter data for 2019-2022
df_filtered = df[(df['date'] >= '2019-01-01') & (df['date'] <= '2022-12-31')]

# Convert date to matplotlib format
df_filtered['date'] = df_filtered['date'].map(mdates.date2num)

# Create the candlestick chart
fig, ax = plt.subplots(figsize=(20, 6))

candlestick_ohlc(ax, df_filtered[['date', 'open', 'high', 'low', 'close']].values,
                 width=0.6, colorup='g', colordown='r')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()

plt.title('Bitcoin Candlestick Chart (2019-2022)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)

# Save the plot as a PDF
plt.savefig('./Module_4/Week_2/Data/bitcoin_candlestick_2019_2022.pdf')

plt.show()

# Create function for Linear Regression Model
def predict(x, w, b):
    return x.dot(w) + b

def gradient(y_hat, y, x):
    loss = y_hat - y
    dw = x.T.dot(loss)/len(y)
    db = np.sum(loss)/len(y)
    cost = np.sum(loss**2)/(2*len(y))
    
    return (dw, db, cost)

def update_weight(w, b, lr, dw, db):
    w_new = w - dw*lr
    b_new = b - db*lr
    return (w_new, b_new)

# Standardizing the prices  
lst_standardized = ['open', 'high', 'low', 'close']
scalar = StandardScaler()

for item in lst_standardized:
    df[f"Standardized_{item.capitalize()}_Prices"] = scalar.fit_transform(df[item].values.reshape(-1,1))

X = df[["Standardized_Open_Prices", "Standardized_High_Prices", "Standardized_Low_Prices"]]
y = df["Standardized_Close_Prices"]

# Split train:test = 7:3
x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    shuffle=True
)

def linear_regression_vectorized(x, y, learning_rate=0.01, num_iterations=200):
    _, n_features = x.shape
    w = np.zeros(n_features) # Initialize weights
    b = 0 # Initialize bias
    losses = []
    
    for _ in range (num_iterations):
        y_hat = predict(x, w, b)
        dw, db, cost = gradient(y_hat, y, x)
        w, b = update_weight(w, b, learning_rate, dw, db)
        losses.append(cost)
        
    return w, b, losses

w, b, losses = linear_regression_vectorized (x_train.values, y_train.values,
                                               learning_rate=0.01, num_iterations=200)
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Function during Gradient Descent')
plt.show()

# Evaluate
# Make predictions on the test set
y_pred = predict(x_test, w, b)

# Calculate RMSE
rmse = np.sqrt(np.mean((y_pred - y_test)**2))

# Calculate MAE
mae = np.mean(np.abs(y_pred - y_test))

# Calculate MAPE
mape = np.mean(np.abs((y_test - y_pred)/y_test))*100

# Calculate R-squared on training data
y_train_pred = predict(x_train, w, b)
train_accuracy = r2_score(y_train, y_train_pred)

# Calculate R-squared on testing data
test_accuracy = r2_score(y_test, y_pred)

print("Root Mean Square Error (RMSE):", round(rmse, 4)) # Root Mean Square Error (RMSE): 0.0293
print("Mean Absolute Error (MAE):", round(mae, 4)) # Mean Absolute Error (MAE): 0.0138
print("Training Accuracy (R-squared):", round(train_accuracy, 4)) # Training Accuracy (R-squared): 0.9989
print("Testing Accuracy (R-squared):", round(test_accuracy, 4)) # Testing Accuracy (R-squared): 0.9991

# Filter data for 2019-01-01 to 2019-04-01
df_2019_q1 = df[(df['date'] >= '2019-01-01') & (df['date'] <= '2019-04-01')]

# Prepare X and y for prediction
x_2019_q1 = df_2019_q1[lst_standardized[:3]]
y_2019_q1_actual = df_2019_q1[lst_standardized[3]]

y_2019_q1_pred = predict(x_2019_q1, w, b)

# Create a plot
plt.figure(figsize=(12, 6))
plt.plot(df_2019_q1['date'], y_2019_q1_actual, label='Actual Close Price', marker='o')
plt.plot(df_2019_q1['date'], y_2019_q1_pred, label='Predicted Close Price', marker='x')
plt.title('Actual vs. Predicted Bitcoin Close Price (01/01/2019 - 04/01/2019)')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()