# -*- coding: utf-8 -*-
"""
Created on March 3, 2025

@author: Leslie Ngonidzashe Kaziwa

Project: Car Price Prediction Using Machine Learning
"""
# import modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# load the dataset
dataset = pd.read_csv('CarPrice.txt')
dataframe = pd.DataFrame(dataset)

# explore the data
print(dataframe.head())

# check for missing values
print(dataframe.isnull().sum())

# convert categorical variables to dummy variables
dataframe = pd.get_dummies(dataframe, drop_first=True)
dataframe.columns = dataframe.columns.str.strip()

# display the transformed data
print(dataframe.head())

# drop non-relevant columns
dataframe = dataframe.drop(columns=['car_ID'], axis=1)
dataframe = dataframe.drop(columns=['CarName'], axis=1, errors='ignore')

# display the new dataframe
print(dataframe.head())

# define the features and target variable
X = dataframe.drop(columns=['price'])
y = dataframe['price']

# split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# check the shape of the datasets
print(f'Training set size: {X_train.shape}')
print(f'Testing set size: {X_test.shape}')

# initialise and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# evaluate the model
y_pred = model.predict(X_test)
print(f'R-squared value: {r2_score(y_test, y_pred)}')

# random forest regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# evaluate the model
rf_y_pred = rf_model.predict(X_test)
print(f'Random Forest R-squared value: {r2_score(y_test, rf_y_pred)}')

# calculate mae and mse
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
