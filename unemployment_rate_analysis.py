# -*- coding: utf-8 -*-
"""
Created on March 2, 2025

@author: Leslie Ngonidzashe Kaziwa

Project: Unemployment Rate Analysis
"""

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# load dataset
dataset = pd.read_csv('Unemployment in India.csv')
dataframe = pd.DataFrame(dataset)

# clean the data
dataframe.columns = dataframe.columns.str.strip()
dataframe = dataframe.dropna(how='all')

# get summary statistics
summary_stats = dataframe['Estimated Unemployment Rate (%)'].describe()
print(summary_stats)

# mean unemployment rate for each region
region_unemployment = dataframe.groupby('Region')['Estimated Unemployment Rate (%)'].mean()
print(region_unemployment)

# convert date column to a datetime object
dataframe['Date'] = pd.to_datetime(dataframe['Date'], dayfirst=True, errors='coerce')

# plot unemployment rate over time
plt.figure(figsize=(10,5))
plt.plot(dataframe['Date'], dataframe['Estimated Unemployment Rate (%)'], label='Unemployment Rate')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.title('Unemployment Rate Over Time')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# correlation between the number of employed and unemployment rate
correlation = dataframe['Estimated Employed'].corr(dataframe['Estimated Unemployment Rate (%)'])
print(f'Correlation between employment and unemployment rate: {correlation}')

# extract month from date column
dataframe['Month'] = dataframe['Date'].dt.month

# group by month and calculate the average unemployment rate for each month
monthly_unemployment = dataframe.groupby('Month')['Estimated Unemployment Rate (%)'].mean()

# plot the seasonal trend
plt.figure(figsize=(10,5))
monthly_unemployment.plot(kind='bar')
plt.xlabel('Month')
plt.ylabel('Average Unemployment Rate (%)')
plt.title('Average Unemployment Rate by Month')
plt.xticks(rotation=45)
plt.show()

# predictive analysis
X = dataframe['Date'].map(lambda x: x.toordinal() if pd.notnull(x) else None).values.reshape(-1,1)
y = dataframe['Estimated Unemployment Rate (%)']

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model
model = LinearRegression()
model.fit(X_train, y_train)

# predict unemployment rate for future dates
y_pred = model.predict(X_test)
