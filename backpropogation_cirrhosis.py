# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:35:02 2024

@author: Jungyu Lee, 301236221

Assignment 2 Exercise 1
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

cirrhosis_dataset = pd.read_csv('cirrhosis.csv')

cirrhosis_dataset.isnull().sum()

# drop where drug is missing
cirrhosis_dataset = cirrhosis_dataset.iloc[:312]

# drop 'ID' column
cirrhosis_dataset.drop(['ID'], axis=1, inplace=True)
X = cirrhosis_dataset.drop(['Stage'], axis=1)  
y = cirrhosis_dataset['Stage']

cirrhosis_dataset.dtypes

numeric_X = X.select_dtypes(include=['int64', 'float64']).columns
categorical_X = X.select_dtypes(include=['object']).columns

numeric_X_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder())
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_X_transformer, numeric_X),
        ('cat', categorical_transformer, categorical_X)
    ])

X_processed = preprocessor.fit_transform(X)
y_processed = OneHotEncoder(sparse=False).fit_transform(np.array(y).reshape(-1, 1))


X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=21)

# neural network structure
input_size = X_train.shape[1]
hidden_layer_size = 6
output_size = y_train.shape[1]

# weight initialization
W1 = np.random.uniform(-1, 1, (input_size, hidden_layer_size))
W2 = np.random.uniform(-1, 1, (hidden_layer_size, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def predict(X, W1, W2):
    hidden_layer_input = np.dot(X, W1)
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_input = np.dot(hidden_layer_output, W2)
    final_output = sigmoid(final_input)
    
    predictions = np.argmax(final_output, axis=1)
    return predictions


def backpropagation(X, y, W1, W2, lr, epochs, batch_size):
    for epoch in range(epochs):
        # shuffle the dataset
        permutation = np.random.permutation(X.shape[0])
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]
        
        for i in range(0, X.shape[0], batch_size):
            X_mini = X_shuffled[i:i+batch_size]
            y_mini = y_shuffled[i:i+batch_size]
            
            # forward propagation
            hidden_layer_input = np.dot(X_mini, W1)
            hidden_layer_output = sigmoid(hidden_layer_input)
            final_input = np.dot(hidden_layer_output, W2)
            final_output = sigmoid(final_input)
            
            error = y_mini - final_output
            
            # backward propagation
            delta_output = error * sigmoid_derivative(final_output)
            dW2 = np.dot(hidden_layer_output.T, delta_output) / batch_size  
            error_hidden_layer = np.dot(delta_output, W2.T) * sigmoid_derivative(hidden_layer_output)
            dW1 = np.dot(X_mini.T, error_hidden_layer) / batch_size
            
            W2 += lr * dW2
            W1 += lr * dW1
        
        train_pred = predict(X, W1, W2)  
        train_labels = np.argmax(y, axis=1)  
        train_accuracy = np.mean(train_pred == train_labels)
        print(f'Epoch {epoch+1}/{epochs}, Training Accuracy: {train_accuracy:.4f}')
        
    return W1, W2

# hyperparameters
lr = 0.1
epochs = 1000
batch_size=32

W1, W2 = backpropagation(X_train, y_train, W1, W2, lr, epochs, batch_size)

y_pred = predict(X_test, W1, W2)

y_test_labels = np.argmax(y_test, axis=1)

test_accuracy = np.mean(y_pred == y_test_labels)
print(f"Test Accuracy: {test_accuracy:.4f}")
