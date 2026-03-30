"""
perceptron_classifier.py - Perceptron Classifier from Scratch
Assignment 1, Question 2 Part 1
Author: Chris Manlove

Uses the Iris flower dataset from sklearn to classify species
using a Perceptron implemented from scratch with NumPy.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════
# Perceptron implementation from scratch
#
# Uses the update rule from the lecture slides (slide 39):
#   y_hat = phi(w^T * x)
#   w_{t+1} = w_t - alpha * (y_hat - y) * x
#
# Activation function: +1 if sum >= 0, else -1
# Labels: +1 and -1 (as shown in lecture)
#
# For multi-class (3 Iris species), we use one-vs-all:
# one perceptron per class, each trained to distinguish
# its class (+1) from all others (-1).
# ══════════════════════════════════════════════════════════════════

class Perceptron:
    def __init__(self, n_features, learning_rate=0.01, n_epochs=100):
        # Initial weights: small random values, bias: 0.0
        self.weights = np.random.RandomState(RANDOM_STATE).randn(n_features) * 0.01
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def activation(self, z):
        # Step function: returns +1 if z >= 0, else -1 (as in lecture slide 39)
        return np.where(z >= 0, 1, -1)

    def predict(self, x):
        # Compute w^T * x + bias, then apply activation
        z = x @ self.weights + self.bias
        return self.activation(z)

    def raw_score(self, x):
        # Raw output before activation (used for one-vs-all tie-breaking)
        return x @ self.weights + self.bias

    def fit(self, x, y):
        # Perceptron learning rule (lecture slide 39):
        # w_{t+1} = w_t - alpha * (y_hat - y) * x
        for epoch in range(self.n_epochs):
            y_pred = self.predict(x)
            errors = y_pred - y  # (y_hat - y)
            # Update weights: w = w - alpha * (y_hat - y) * x
            self.weights -= self.learning_rate * (x.T @ errors)
            self.bias -= self.learning_rate * errors.sum()


class OneVsAllPerceptron:
    """
    Multi-class classification using one perceptron per class.
    Each perceptron learns: "is this class i (+1) or not (-1)?"
    Final prediction picks the class whose perceptron gives the highest raw score.
    """
    def __init__(self, n_features, n_classes, learning_rate=0.01, n_epochs=100):
        self.perceptrons = [
            Perceptron(n_features, learning_rate, n_epochs)
            for _ in range(n_classes)
        ]
        self.n_classes = n_classes

    def fit(self, x, y):
        for i, p in enumerate(self.perceptrons):
            # Convert labels to +1/-1: +1 if class == i, else -1
            binary_y = np.where(y == i, 1, -1)
            p.fit(x, binary_y)

    def predict(self, x):
        # Get raw scores from each perceptron, pick the highest
        scores = np.column_stack([p.raw_score(x) for p in self.perceptrons])
        return np.argmax(scores, axis=1)
    


# ══════════════════════════════════════════════════════════════════
# Load and prepare the Iris dataset
# ══════════════════════════════════════════════════════════════════
iris = load_iris()
x = iris.data       # shape (150, 4): sepal length, sepal width, petal length, petal width
y = iris.target     # shape (150,): class labels 0, 1, 2

# (a) Split the dataset into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=RANDOM_STATE
)

# Standardize the data (important for neural networks, as shown in lecture slide 52)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(f"Training samples: {len(x_train)}")
print(f"Testing samples:  {len(x_test)}")
print(f"Number of features: {x_train.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")


# ══════════════════════════════════════════════════════════════════
# (b) Train the Perceptron from scratch and compute accuracy
# ══════════════════════════════════════════════════════════════════

# Hyperparameters
learning_rate = 0.01
n_epochs = 100

print(f"\n--- Perceptron Hyperparameters ---")
print(f"Initial weights: small random values (normal distribution * 0.01)")
print(f"Initial bias: 0.0")
print(f"Learning rate: {learning_rate}")
print(f"Number of epochs: {n_epochs}")
print(f"Activation function: step function (+1 if z >= 0, else -1)")

model = OneVsAllPerceptron(
    n_features=x_train.shape[1],
    n_classes=len(np.unique(y)),
    learning_rate=learning_rate,
    n_epochs=n_epochs,
)
model.fit(x_train, y_train)

# Compute classification accuracy = (correct predictions / total) * 100
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_accuracy = np.mean(y_train_pred == y_train) * 100
test_accuracy = np.mean(y_test_pred == y_test) * 100

print(f"\n--- Perceptron Results ---")
print(f"Training accuracy: {train_accuracy:.2f}%")
print(f"Testing accuracy:  {test_accuracy:.2f}%")
