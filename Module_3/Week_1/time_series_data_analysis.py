import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#########################
# Load data
#########################

dataset_path = './Module_3/Week_1/Data/opsd_germany_daily.csv' # Ensure this path is correct and points to your opsd_germany_daily.csv file

# Read data from .csv file
opsd_daily = pd.read_csv(dataset_path)

print("Some info in DataFrame:")
print(opsd_daily.shape)
print(opsd_daily.dtypes)
print(opsd_daily.head(3))

# Set the Date column to the index column
opsd_daily = opsd_daily.set_index('Date')
print(f"\nNew DataFrame: \n{opsd_daily.head(3)}")

# Repeat the file loading step, specify the column that will be indexed right from the time of calling the function, 
# and also create additional column numbers Year, Month, and Weekday extracted from the Date column
opsd_daily = pd.read_csv(dataset_path, index_col=0, parse_dates=True)

## Add columns with year , month , and weekday name
opsd_daily['Year'] = opsd_daily.index.year
opsd_daily['Month'] = opsd_daily.index.month
opsd_daily['Weekday Name'] = opsd_daily.index.day_name()
## Display a random sampling of 5 rows
print(f"\nRandom sampling: \n{opsd_daily.sample(5, random_state=0)}")


#########################
# Time-based indexing
#########################

# Access data from date 2014-01-20 to 2014-01-22:
print(f"\nSlicing: \n{opsd_daily.loc['2014-01-20':'2014-01-22']}")

# Partial-string indexing
print(f"\nPartial-string indexing: \n{opsd_daily.loc['2012-02']}")


#########################
# Visualizing time series data
#########################

# Plot data of the Consumption column
def plot_consumption_column():
    plt.figure(num="Consumption Data")
    sns.set_theme(rc={'figure.figsize':(11, 4)})
    opsd_daily['Consumption'].plot(linewidth=0.5)
    plt.show() 
# plot_consumption_column()

# Plot several columns of data at once into separate graphs
def plot_columns():
    cols_plot = ['Consumption', 'Solar', 'Wind']
    axes = opsd_daily[cols_plot].plot(marker='.', alpha=0.5, linestyle='None',
                                    figsize=(11, 9), subplots=True)
    for ax in axes:
        ax.set_ylabel('Daily Totals (Gwh)')
    plt.show()
# plot_columns()


#########################
# Seasonality
#########################
def seasonality():
    _, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for name, ax in zip(['Consumption', 'Solar', 'Wind'], axes):
        sns.boxplot(data=opsd_daily, x='Month', y=name, ax=ax)
        ax.set_ylabel('GWh')
        ax.set_title(name)
        # Remove the automatic x- axis label from all but the bottom subplot
        if ax != axes[-1]:
            ax.set_xlabel('')
    plt.show()
# seasonality()


#########################
# Frequencies
#########################

# Create a time list with frequency by day
print("Date Range:\n", pd.date_range('1998-03-10', '1998-03 -15', freq='D'))

# To select an arbitrary sequence of date / time values from a pandas time series ,
# we need to use a DatetimeIndex , rather than simply a list of date / time strings
times_sample = pd.to_datetime(['2013-02-03', '2013-02-06', '2013-02-08'])
# Select the specified dates and just the Consumption column
consum_sample = opsd_daily.loc[times_sample, ['Consumption']]. copy ()
print("\nConsumption sample:\n",consum_sample)

# Convert the data to daily frequency , without filling any missings
consum_freq = consum_sample.asfreq('D')
# Create a column with missings forward filled
consum_freq['Consumption - Forward Fill'] = consum_sample.asfreq('D', method='ffill')
print("\nFfill:\n",consum_freq)


#########################
# Resampling
#########################

# Specify the data columns we want to include (i.e. exclude Year, Month, Weekday, Name)
wind_solar_column = 'Wind+Solar'
data_columns = ['Consumption', 'Wind', 'Solar', wind_solar_column]
# Resample to weekly frequency , aggregating with mean
opsd_weekly_mean = opsd_daily[data_columns].resample('W').mean()
print("\nDownsampling:")
print(opsd_weekly_mean.head(3))
print("Daily shape:", opsd_daily.shape[0])
print("Weekly shape:", opsd_weekly_mean.shape[0])

# Visualize daily và weekly time series of the Solar column in 6 months
def plot_daily_weekly_solar():
    # Start and end of the date range to extract
    start, end = '2017-01', '2017-06'

    # Plot daily and weekly resampled time series together
    _, ax = plt.subplots(num="Solar Production Comparison")
    ax.plot(opsd_daily.loc[start:end, 'Solar'],
            marker='.', linestyle='-', linewidth=0.5, label='Daily')
    ax.plot(opsd_weekly_mean.loc[start:end, 'Solar'],
            marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')
    ax.set_ylabel('Solar Production (GWh)')
    ax.legend()
    plt.show()
# plot_daily_weekly_solar()

# Compute the annual sums, setting the value to NaN for any year which has
# fewer than 360 days of data
opsd_annual = opsd_daily[data_columns].resample('YE').sum(min_count=360)
# The default index of the resampled DataFrame is the last day of each year,
# ('2006-12-31', '2007-12-31', etc.) so to make life easier, set the index
# to the year component
opsd_annual = opsd_annual.set_index(opsd_annual.index.year)
opsd_annual.index.name = 'Year'
# Compute the ratio of Wind+Solar to Consumption
opsd_annual['Wind+Solar/Consumption'] = opsd_annual['Wind+Solar'] / opsd_annual['Consumption']
print("\nAnnual DataFrame:\n", opsd_annual.tail(3))

# Plot from 2012 onwards, because there is no solar production data in earlier years
def plot_annual():
    ax = opsd_annual.loc[2012:, 'Wind+Solar/Consumption'].plot.bar(color='C0')

    ax.set_ylabel('Fraction')
    ax.set_ylim(0, 0.3)
    ax.set_title('Wind + Solar Share of Annual Electricity Consumption')

    plt.xticks(rotation=0)
    plt.show()
# plot_annual()


#########################
# Rolling
#########################
# Compute the centered 7-day rolling mean
opsd_7d = opsd_daily[data_columns].rolling(7, center=True).mean()
print("\nRolling:\n",opsd_7d.head(10))


#########################
# Trends
#########################
# The min_periods=360 argument accounts for a few isolated missing days in the
# wind and solar production time series
opsd_365d = opsd_daily[data_columns].rolling(window=365, center=True, min_periods=360).mean()
def plot_365_day_rolling():
    # Plot daily, 7-day rolling mean, and 365-day rolling mean time series
    _, ax = plt.subplots()
    ax.plot(opsd_daily['Consumption'], marker='.', markersize=2, color='0.6', linestyle='None', label='Daily')
    ax.plot(opsd_7d['Consumption'], linewidth=2, label='7-d Rolling Mean')
    ax.plot(opsd_365d['Consumption'], color='0.2', linewidth=3, label='Trend (365-d Rolling Mean)')

    # Set x-ticks to yearly interval and add legend and labels
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel('Consumption (GWh)')
    ax.set_title('Trends in Electricity Consumption')
    plt.show()
# plot_365_day_rolling()

def plot_365_day_rolling_wands():
    # Plot 365-day rolling mean time series of wind and solar power
    _, ax = plt.subplots()

    for nm in ['Wind', 'Solar', wind_solar_column]:
        ax.plot(opsd_365d[nm], label=nm)

    # Set x-ticks to yearly interval, adjust y-axis limits, add legend and labels
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_ylim(0, 400)
    ax.legend()
    ax.set_ylabel('Production (GWh)')
    ax.set_title('Trends in Electricity Production (365-d Rolling Means)')
    plt.show()
# plot_365_day_rolling_wands()