"""
multiple_LR.py - Multiple Linear Regression
Assignment 1, Question 1 Part 2
Author: Chris Manlove

Uses GasProperties.csv to learn a regression model predicting gas quality (Idx)
based on chemical properties (T, P, TC, SV).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 0

# ── Load the dataset ──────────────────────────────────────────────
df = pd.read_csv("GasProperties.csv")

# Features: T, P, TC, SV
# Target: Idx (gas quality index)
feature_cols = ["T", "P", "TC", "SV"]
x = df[feature_cols].values
y = df["Idx"].values


# ══════════════════════════════════════════════════════════════════
# (a) Split the dataset into training set (80%) and testing set (20%)
# ══════════════════════════════════════════════════════════════════
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=RANDOM_STATE
)

print(f"Training samples: {len(x_train)}")
print(f"Testing samples:  {len(x_test)}")

# ══════════════════════════════════════════════════════════════════
# (b) Implement least squares from scratch using NumPy
#     Computes w_hat = (x^T x)^{-1} x^T y
#     This is the closed-form solution that minimizes
#     J(w) = (1/N) * sum( (y_i - w^T x_i)^2 )
# ══════════════════════════════════════════════════════════════════

# We need to prepend a column of 1s to x so that the first weight
# acts as the intercept (bias) term. This is called the "design matrix."
# Example: if x = [[t, p, tc, sv]], then X_aug = [[1, t, p, tc, sv]]
# so w_hat = [w0, w1, w2, w3, w4] and prediction = w0 + w1*t + w2*p + ...
x_train_aug = np.column_stack([np.ones(len(x_train)), x_train])
x_test_aug = np.column_stack([np.ones(len(x_test)), x_test])

# w_hat = (x^T x)^{-1} x^T y
xtx = x_train_aug.T @ x_train_aug       # matrix multiply: x^T * x
xtx_inv = np.linalg.inv(xtx)            # invert the matrix
xty = x_train_aug.T @ y_train           # x^T * y
w_hat = xtx_inv @ xty                   # final weight vector

print(f"\nWeight vector (w_hat): {w_hat}")
print(f"  w0 (intercept) = {w_hat[0]:.6f}")
for i, col in enumerate(feature_cols):
    print(f"  w{i+1} ({col})        = {w_hat[i+1]:.6f}")

# ══════════════════════════════════════════════════════════════════
# (c) Compute training and testing RMSE
#     Predictions are computed as y_hat = X_aug @ w_hat
#     (matrix-vector multiply: each row dotted with the weight vector)
# ══════════════════════════════════════════════════════════════════
y_train_pred = x_train_aug @ w_hat
y_test_pred = x_test_aug @ w_hat

rmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
rmse_test = np.sqrt(np.mean((y_test - y_test_pred) ** 2))

print(f"\n--- Before Normalization ---")
print(f"Training RMSE: {rmse_train:.4f}")
print(f"Testing RMSE:  {rmse_test:.4f}")

# ══════════════════════════════════════════════════════════════════
# (d) Normalize T, P, TC, SV using z-score: z = (x - mean) / std
#     Then exclude any row where ANY feature has |z| > 2 (outlier).
#     Save the result as GasProperties_norm.csv.
# ══════════════════════════════════════════════════════════════════
df_norm = df.copy()

# Compute z-scores for each feature column
for col in feature_cols:
    mean = df_norm[col].mean()
    std = df_norm[col].std()
    df_norm[col] = (df_norm[col] - mean) / std

# Remove rows where any feature is more than 2 standard deviations away
# abs(z) > 2 means the value is an outlier
outlier_mask = (df_norm[feature_cols].abs() > 2).any(axis=1)
rows_before = len(df_norm)
df_norm = df_norm[~outlier_mask]
rows_after = len(df_norm)

print(f"\n--- Normalization ---")
print(f"Rows before outlier removal: {rows_before}")
print(f"Rows after outlier removal:  {rows_after}")
print(f"Outliers removed:            {rows_before - rows_after}")

df_norm.to_csv("GasProperties_norm.csv", index=False)
print("Saved normalized dataset to GasProperties_norm.csv")

# ══════════════════════════════════════════════════════════════════
# (e) Retrain linear regression on normalized dataset using
#     the same least squares method: w_hat = (x^T x)^{-1} x^T y
# ══════════════════════════════════════════════════════════════════
x_norm = df_norm[feature_cols].values
y_norm = df_norm["Idx"].values

x_norm_train, x_norm_test, y_norm_train, y_norm_test = train_test_split(
    x_norm, y_norm, test_size=0.2, random_state=RANDOM_STATE
)

# Augment with bias column of 1s
x_norm_train_aug = np.column_stack([np.ones(len(x_norm_train)), x_norm_train])
x_norm_test_aug = np.column_stack([np.ones(len(x_norm_test)), x_norm_test])

# Compute weights
xtx_norm = x_norm_train_aug.T @ x_norm_train_aug
xtx_norm_inv = np.linalg.inv(xtx_norm)
xty_norm = x_norm_train_aug.T @ y_norm_train
w_hat_norm = xtx_norm_inv @ xty_norm

print(f"\nNormalized weight vector: {w_hat_norm}")

# Compute RMSE on normalized data
y_norm_train_pred = x_norm_train_aug @ w_hat_norm
y_norm_test_pred = x_norm_test_aug @ w_hat_norm

rmse_norm_train = np.sqrt(np.mean((y_norm_train - y_norm_train_pred) ** 2))
rmse_norm_test = np.sqrt(np.mean((y_norm_test - y_norm_test_pred) ** 2))

print(f"\n--- After Normalization ---")
print(f"Training RMSE: {rmse_norm_train:.4f}")
print(f"Testing RMSE:  {rmse_norm_test:.4f}")

# ══════════════════════════════════════════════════════════════════
# (f) Compare RMSE before and after normalization
# ══════════════════════════════════════════════════════════════════
print(f"\n--- Comparison ---")
print(f"{'':20s} {'Before':>12s} {'After':>12s}")
print(f"{'Training RMSE':20s} {rmse_train:12.4f} {rmse_norm_train:12.4f}")
print(f"{'Testing RMSE':20s} {rmse_test:12.4f} {rmse_norm_test:12.4f}")

if rmse_norm_test < rmse_test:
    print("\nNormalization improved accuracy.")
    print("Possible reason: Removing outliers eliminated extreme values that were")
    print("distorting the regression fit. The outliers pulled the weight vector")
    print("away from the true relationship, increasing error on typical data points.")
else:
    print("\nNormalization did not improve accuracy.")
    print("Possible reason: The original features may already have been on similar")
    print("scales, so normalization did not change the fit significantly. However,")
    print("outlier removal reduced the dataset size, which may have offset any gain.")