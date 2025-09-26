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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.cost_history = []

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.W) + self.b
            y_pred = sigmoid(linear_model)

            cost = -(1/self.m) * np.sum(y*np.log(y_pred+1e-9) + (1-y)*np.log(1-y_pred+1e-9))
            self.cost_history.append(cost)

            dw = (1/self.m) * np.dot(X.T, (y_pred - y))
            db = (1/self.m) * np.sum(y_pred - y)

            self.W -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return sigmoid(np.dot(X, self.W) + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
X = np.array([
    [8.5, 120], [6.2, 100], [7.8, 110], [5.5, 85],
    [9.0, 130], [6.0, 95], [7.0, 105], [8.8, 125]
])
y = np.array([1, 0, 1, 0, 1, 0, 1, 1])

X = (X - X.mean(axis=0)) / X.std(axis=0)

model = LogisticRegressionGD(lr=0.1, epochs=2000)
model.fit(X, y)

y_pred = model.predict(X)
print("Predictions:", y_pred)
print("Actual     :", y)
print("Accuracy   :", accuracy_score(y, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))

plt.plot(model.cost_history)
plt.title("Cost Function Convergence")
plt.xlabel("Epochs")
plt.ylabel("Cost (Log Loss)")
plt.show()

plt.scatter(X[:,0], X[:,1], c=y, cmap="bwr", edgecolors="k")
x1 = np.linspace(X[:,0].min(), X[:,0].max(), 100)
x2 = -(model.W[0]*x1 + model.b) / model.W[1]  # boundary equation
plt.plot(x1, x2, color="green", label="Decision Boundary")
plt.legend()
plt.title("Decision Boundary for Logistic Regression (GD)")
plt.xlabel("Feature 1 (scaled CGPA)")
plt.ylabel("Feature 2 (scaled IQ)")
plt.show()

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

