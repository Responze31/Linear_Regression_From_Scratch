import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for _ in range(self.n_iterations):
            y_pred = X.dot(self.weights) + self.bias
            dw = (1/n_samples) * X.T.dot(y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

    def predict(self, X):
        return X.dot(self.weights) + self.bias

if __name__ == "__main__":
    # 1. Create a synthetic dataset
    X, y = make_regression(
        n_samples=100,
        n_features=1,
        noise=20,
        random_state=42
    )

    # 2. Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # 3. Train your custom model
    model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)

    # 4. Predict on the test set
    y_pred = model.predict(X_test)

    # 5. Compute metrics
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print("Performance on test set:")
    print(f"  MSE : {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  R²  : {r2:.4f}")

    # 6. Plot test points and fitted line
    plt.scatter(X_test, y_test, label="Test data")
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, label="Fitted line")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Custom Linear Regression Fit")
    plt.legend()
    plt.show()