# Linear Regression from Scratch

This repository implements a simple Linear Regression model using pure Python and NumPy, trained via Gradient Descent. The goal is to illustrate the underlying mathematics and algorithmic steps without relying on high-level libraries.

## Table of Contents
- [Overview]
- [Mathematical Background]
  - [Linear Hypothesis]
  - [Cost Function (MSE)]
  - [Gradient Descent]
- [Implementation Details]
  - [Class Structure]
  - [Training (fit)]
  - [Prediction]
- [Usage Example]
- [Performance Metrics]
- [Visualization]
- [Requirements]
- [License]

## Overview

We generate a synthetic dataset with one feature and a continuous target, split it into training and test sets, then fit a custom Linear Regression model using Gradient Descent.

Key steps:
- Generate data: `X, y = make_regression(n_samples=100, n_features=1, noise=20)`
- Split: 80% train, 20% test
- Fit model via Gradient Descent
- Evaluate using MSE, RMSE, MAE, R²
- Plot the resulting fit

## Mathematical Background

### Linear Hypothesis
We assume a linear relationship between input x and output y:

y = wx + b

where:
- w (weight) is the slope of the line
- b (bias) is the intercept

In vector form for multiple features X ∈ ℝ^(n×d):

y = Xw + b

### Cost Function (MSE)
We measure the quality of our hypothesis using the Mean Squared Error (MSE):

J(w,b) = (1/2n) * Σ(y_pred - y_actual)²

The factor 1/2 simplifies derivative expressions.

### Gradient Descent
We optimize (w, b) by iteratively updating along the negative gradient of the cost:

Weight gradient:
∂J/∂w = (1/n) * X^T * (Xw + b - y)

Bias gradient:
∂J/∂b = (1/n) * Σ(y_pred - y)

Update rules for learning rate α:
w = w - α * ∂J/∂w
b = b - α * ∂J/∂b

Repeat for T iterations.

## Implementation Details

### Class Structure
```python
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        # Initialize hyperparameters and placeholders
        self.learning_rate = learning_rate  # α
        self.n_iterations = n_iterations    # T
        self.weights = None                 # w
        self.bias = 0.0                     # b
```

### Training (fit)
```python
def fit(self, X, y):
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0.0
    
    for _ in range(self.n_iterations):
        y_pred = X.dot(self.weights) + self.bias
        
        # Compute gradients
        dw = (1/n_samples) * X.T.dot(y_pred - y)
        db = (1/n_samples) * np.sum(y_pred - y)
        
        # Update parameters
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
```

- Computes gradients
- Applies update rules each iteration

### Prediction (predict)
```python
def predict(self, X):
    return X.dot(self.weights) + self.bias
```

Returns predicted values given features.

## Usage Example

```python
# Import dependencies
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Test data')
plt.scatter(X_test, y_pred, color='red', label='Predictions')

# Plot regression line
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, color='green', label='Regression Line')

plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression from Scratch')
plt.legend()
plt.show()
```

This will:
- Generate data
- Train the model
- Print metrics: MSE, RMSE, MAE, R²
- Display a plot of test data vs. fitted line

## Performance Metrics

- Mean Squared Error (MSE): (1/n)∑(y - ŷ)²
- Root MSE (RMSE): √MSE
- Mean Absolute Error (MAE): (1/n)∑|y - ŷ|
- Coefficient of Determination (R²): Measures variance explained

## Visualization

We plot the test points (x^(i), y^(i)) and the fitted line:

```python
plt.scatter(X_test, y_test, label="Test data")
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, label="Fitted line")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Custom Linear Regression Fit")
plt.legend()
plt.show()
```

## Requirements

- Python 3.6+
- NumPy
- scikit-learn
- Matplotlib

Install via:
```
pip install numpy scikit-learn matplotlib
```

## License

This project is released under the MIT License. Feel free to use and modify!
