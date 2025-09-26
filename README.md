# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries: Import NumPy, pandas, and StandardScaler for numerical operations, data handling, and feature scaling, respectively.

2.Define the Linear Regression Function: Create a linear regression function using gradient descent to iteratively update parameters, minimizing the difference between predicted and actual values.

3.Load and Preprocess the Data: Load the dataset, extract features and target variable, and standardize both using StandardScaler for consistent model training.

4.Perform Linear Regression: Apply the defined linear regression function to the scaled features and target variable, obtaining optimal parameters for the model.

5.Make Predictions on New Data: Prepare new data, scale it, and use the trained model to predict the target variable, transforming predictions back to the original scale.

6.Print the Predicted Value

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: GEETHU R
RegisterNumber:  212224040089
*/
```
~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\admin\Downloads\Placement_Data (1).csv")

dataset = dataset.drop("sl_no", axis=1)
dataset = dataset.drop("salary", axis=1)

categorical_cols = [
    "gender", "ssc_b", "hsc_b", "degree_t",
    "workex", "specialisation", "status", "hsc_s"
]
for col in categorical_cols:
    dataset[col] = dataset[col].astype("category")

for col in categorical_cols:
    dataset[col] = dataset[col].cat.codes

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X = np.hstack((np.ones((X.shape[0], 1)), X))

theta = np.random.randn(X.shape[1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))

def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)

def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

y_pred = predict(theta, X)

accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)

print("Actual Y values:")
print(y)

xnew = np.array([[1, 0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print("Prediction for new input 1:", y_prednew)

xnew2 = np.array([[1, 0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew2 = predict(theta, xnew2)
print("Prediction for new input 2:", y_prednew2)

~~~

## Output:
Read the file and display


<img width="1217" height="422" alt="0" src="https://github.com/user-attachments/assets/1c2b4417-901d-42d1-98e5-574cb22c4712" />



Printing accuracy


<img width="400" height="45" alt="1c" src="https://github.com/user-attachments/assets/f3537bfd-7b82-4438-aaf0-0ef5579e4068" />

Printing Y


<img width="767" height="155" alt="2c" src="https://github.com/user-attachments/assets/da7dc842-b9fc-4628-92c3-4c40a6a89719" />

Printing y_prednew


<img width="351" height="63" alt="3c" src="https://github.com/user-attachments/assets/bf5fd31c-1020-4db1-ae79-87327b76350a" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

