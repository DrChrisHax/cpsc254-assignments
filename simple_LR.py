"""
simple_LR.py - Simple Linear Regression
Assignment 1, Question 1 Part 1
Author: Chris Manlove

Uses study_data.csv (Hours vs Score) to fit a linear regression model.
(a) 80/20 train/test split
(b) Fit using sklearn LinearRegression
(c) Compute training and testing RMSE
(d) Print the learned polynomial: y = w0 + w1*x
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("study_data.csv")

x = df["Hours"].values.reshape(-1, 1)
y = df["Score"].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


print(f"Training samples: {len(x_train)}")
print(f"Testing samples:  {len(x_test)}")

model = LinearRegression()
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

rmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
rmse_test = np.sqrt(np.mean((y_test - y_test_pred) ** 2))

print(f"\nTraining RMSE: {rmse_train:.4f}")
print(f"Testing RMSE:  {rmse_test:.4f}")

w0 = model.intercept_
w1 = model.coef_[0]

print(f"\nLearned function: y = {w0:.4f} + {w1:.4f} * x")