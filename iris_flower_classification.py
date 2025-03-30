# -*- coding: utf-8 -*-
"""
Created on March 2, 2025

@author: Leslie Ngonidzashe Kaziwa

Project: Iris Flower Classification using Machine Learning
"""

# import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load dataset
from sklearn.datasets import load_iris
iris = load_iris()

# convert dataset into dataframe
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data['species'] = iris.target

# explore dataset before training the model
print(iris_data.head())

# preprocess the data and divide it into training and test datasets
X = iris_data.drop('species', axis=1)
y = iris_data['species']

# split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalise the data for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# training a learning model using the support vector machine model to classify the flowers
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# make predictions and evaluate the model
y_pred = model.predict(X_test)

# evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:2f}%')

# generate classification report
print(classification_report(y_test, y_pred))

# confusion matrix to visualise performance
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
