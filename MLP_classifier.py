"""
MLP_classifier.py - The Multilayer Perceptron Classifier
Assignment 1, Question 2 Part 2
Author: Sinan Abdul-Hafiz

Perform a classification using the MLPClassifier in scikit-learn with only one hidden layer with 3
neurons and one output neuron.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════
# Load and prepare the Iris dataset
# (Same split and scaling as perceptron_classifier.py)
# ══════════════════════════════════════════════════════════════════
iris = load_iris()
x = iris.data       # shape (150, 4): sepal length, sepal width, petal length, petal width
y = iris.target     # shape (150,): class labels 0, 1, 2

# (a) Split the dataset into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=RANDOM_STATE
)

# Standardize the data (important for neural networks)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(f"Training samples:   {len(x_train)}")
print(f"Testing samples:    {len(x_test)}")
print(f"Number of features: {x_train.shape[1]}")
print(f"Number of classes:  {len(np.unique(y))}")


# ══════════════════════════════════════════════════════════════════
# (a) Simple MLP: one hidden layer with 3 neurons, one output neuron
#
# Architecture: 4 inputs → [3 hidden] → 3 outputs (one per class)
#
# Note: scikit-learn's MLPClassifier uses a softmax output layer
# for multi-class problems, so "one output neuron" maps to 3 output
# nodes internally (one per class). The hidden_layer_sizes=(3,)
# argument specifies the single hidden layer with 3 neurons.
# ══════════════════════════════════════════════════════════════════

# Hyperparameters
hidden_layers_simple  = (3,)       # one hidden layer with 3 neurons
activation_fn         = 'relu'     # ReLU activation for hidden layer
solver                = 'sgd'      # stochastic gradient descent (closest to lecture perceptron)
learning_rate_init    = 0.01       # initial learning rate
max_iter              = 1000       # number of epochs
batch_size            = 32         # mini-batch size
momentum              = 0.0        # no momentum (pure SGD)
alpha                 = 0.0001     # L2 regularization term

print(f"\n{'═'*60}")
print(f"  Part (a): Simple MLP — 1 hidden layer, 3 neurons")
print(f"{'═'*60}")
print(f"Architecture:       4 → 3 → 3 (input → hidden → output)")
print(f"Activation:         {activation_fn} (hidden), softmax (output, implicit)")
print(f"Initial weights:    Glorot uniform (sklearn default)")
print(f"Initial bias:       0.0 (sklearn default)")
print(f"Solver:             {solver}")
print(f"Learning rate:      {learning_rate_init}")
print(f"Number of epochs:   {max_iter}")
print(f"Batch size:         {batch_size}")
print(f"Momentum:           {momentum}")
print(f"L2 regularization:  {alpha}")

mlp_simple = MLPClassifier(
    hidden_layer_sizes=hidden_layers_simple,
    activation=activation_fn,
    solver=solver,
    learning_rate_init=learning_rate_init,
    max_iter=max_iter,
    batch_size=batch_size,
    momentum=momentum,
    alpha=alpha,
    random_state=RANDOM_STATE,
)
mlp_simple.fit(x_train, y_train)

train_acc_simple = mlp_simple.score(x_train, y_train) * 100
test_acc_simple  = mlp_simple.score(x_test,  y_test)  * 100

print(f"\n--- Simple MLP Results ---")
print(f"Training accuracy:  {train_acc_simple:.2f}%")
print(f"Testing accuracy:   {test_acc_simple:.2f}%")


# ══════════════════════════════════════════════════════════════════
# (b) Deeper MLP: increased complexity to improve accuracy
#
# Changes made from the simple model:
#   - Added more hidden layers: (64, 32, 16)
#   - Switched to 'adam' optimizer (adaptive learning rate,
#     better convergence than vanilla SGD for deeper networks)
#   - Increased max_iter to allow full convergence
#   - Kept the same activation function (ReLU)
#
# Rationale: The Iris dataset is relatively simple, so even the
# small MLP achieves good accuracy. The deeper network with Adam
# converges more reliably and reaches higher training accuracy,
# demonstrating the effect of increased model capacity.
# ══════════════════════════════════════════════════════════════════

hidden_layers_deep  = (64, 32, 16)  # three hidden layers
solver_deep         = 'adam'        # adaptive optimizer for deeper nets
max_iter_deep       = 2000          # more epochs for deeper network to converge

print(f"\n{'═'*60}")
print(f"  Part (b): Deep MLP — 3 hidden layers (64 → 32 → 16)")
print(f"{'═'*60}")
print(f"Architecture:       4 → 64 → 32 → 16 → 3 (input → hidden layers → output)")
print(f"Activation:         {activation_fn} (hidden), softmax (output, implicit)")
print(f"Initial weights:    Glorot uniform (sklearn default)")
print(f"Initial bias:       0.0 (sklearn default)")
print(f"Solver:             {solver_deep} (adaptive learning rate)")
print(f"Learning rate:      {learning_rate_init} (initial; adam adapts per-parameter)")
print(f"Number of epochs:   {max_iter_deep}")
print(f"Batch size:         {batch_size}")
print(f"L2 regularization:  {alpha}")

mlp_deep = MLPClassifier(
    hidden_layer_sizes=hidden_layers_deep,
    activation=activation_fn,
    solver=solver_deep,
    learning_rate_init=learning_rate_init,
    max_iter=max_iter_deep,
    batch_size=batch_size,
    alpha=alpha,
    random_state=RANDOM_STATE,
)
mlp_deep.fit(x_train, y_train)

train_acc_deep = mlp_deep.score(x_train, y_train) * 100
test_acc_deep  = mlp_deep.score(x_test,  y_test)  * 100

print(f"\n--- Deep MLP Results ---")
print(f"Training accuracy:  {train_acc_deep:.2f}%")
print(f"Testing accuracy:   {test_acc_deep:.2f}%")


# ══════════════════════════════════════════════════════════════════
# Summary comparison
# ══════════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  Summary")
print(f"{'═'*60}")
print(f"{'Model':<30} {'Train Acc':>10} {'Test Acc':>10}")
print(f"{'-'*50}")
print(f"{'Simple MLP (4→3→3)':<30} {train_acc_simple:>9.2f}% {test_acc_simple:>9.2f}%")
print(f"{'Deep MLP (4→64→32→16→3)':<30} {train_acc_deep:>9.2f}% {test_acc_deep:>9.2f}%")
print(f"\nImprovement in test accuracy: {test_acc_deep - test_acc_simple:+.2f}%")

# ══════════════════════════════════════════════════════════════════
# Explanation:
#   The deeper network (4→64→32→16→3) with the Adam optimizer generally
#   achieves higher or equal accuracy compared to the minimal 3-neuron MLP.
#   The wider first hidden layer (64 neurons) gives the model more capacity
#   to learn feature combinations, while the subsequent layers (32, 16)
#   progressively compress the representation toward the 3 output classes.
#   Adam's adaptive per-parameter learning rates help the deeper network
#   converge more reliably than SGD, which can stall with more layers.
#   On the Iris dataset the margin of improvement may be modest because
#   the dataset is linearly separable in most class pairs, but the
#   architectural changes demonstrate the general principle that increased
#   depth and a better optimizer improve classification performance.
# ══════════════════════════════════════════════════════════════════