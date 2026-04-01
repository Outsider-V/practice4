import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap


class BaseRegression:
    def __init__(self, learning_rate=0.0001, n_iters=30000, penalty=None, lambda_param=0.01):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.penalty = penalty
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None

    def fit(self, X, y, method='batch'):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            if method == 'sgd':
                for _ in range(n_samples):
                    idx = np.random.randint(0, n_samples)
                    xi = X[idx:idx+1]
                    yi = y[idx:idx+1]
                    y_pred = self._forward(xi)
                    dw, db = self._compute_gradients(xi, yi, y_pred, 1)
                    self.weights -= self.lr * dw
                    self.bias -= self.lr * db
            else:
                y_pred = self._forward(X)
                dw, db = self._compute_gradients(X, y, y_pred, n_samples)
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

    def _forward(self, X):
        raise NotImplementedError

    def _compute_gradients(self, X, y, y_pred, n):
        dw = (1 / n) * np.dot(X.T, (y_pred - y))
        db = (1 / n) * np.sum(y_pred - y)
        if self.penalty == 'l2':
            dw += (self.lambda_param / n) * self.weights
        elif self.penalty == 'l1':
            dw += (self.lambda_param / n) * np.sign(self.weights)
        return dw, db

    def predict(self, X):
        raise NotImplementedError


class MyOwnLinearRegression(BaseRegression):
    def _forward(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        return self._forward(X)


class LogisticRegressionGD(BaseRegression):
    def __init__(self, learning_rate=0.001, n_iters=1000, penalty=None, lambda_param=0.01):
        super().__init__(learning_rate, n_iters, penalty, lambda_param)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _forward(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X):
        y_predicted = self._forward(X)
        return np.array([1 if i > 0.5 else 0 for i in y_predicted])


def run_my_own_logistic_regression():
    data = pd.read_csv("Social_Network_Ads.csv")
    X_raw = data[["Age", "EstimatedSalary"]].values
    y = data["Purchased"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=0
    )
    X_test_raw = scaler.inverse_transform(X_test)

    model = LogisticRegressionGD(learning_rate=0.01, n_iters=1000, penalty='l2', lambda_param=0.1)
    model.fit(X_train, y_train, method='sgd')

    y_pred = model.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    cmap = ListedColormap(("red", "green"))

    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.linspace(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 300),
                         np.linspace(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 300))
    Z = model.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)
    axes[0].contourf(X1, X2, Z, alpha=0.75, cmap=cmap)
    for i, label in enumerate(np.unique(y_set)):
        axes[0].scatter(X_set[y_set == label, 0], X_set[y_set == label, 1], color=["red", "green"][i], label=label)
    axes[0].set_title("Scaled Features (L2 + SGD)")
    axes[0].legend()

    X_set, y_set = X_test_raw, y_test
    X1, X2 = np.meshgrid(np.linspace(X_set[:, 0].min() - 5, X_set[:, 0].max() + 5, 300),
                         np.linspace(X_set[:, 1].min() - 10000, X_set[:, 1].max() + 10000, 300))
    Z = model.predict(scaler.transform(np.c_[X1.ravel(), X2.ravel()])).reshape(X1.shape)
    axes[1].contourf(X1, X2, Z, alpha=0.75, cmap=cmap)
    for i, label in enumerate(np.unique(y_set)):
        axes[1].scatter(X_set[y_set == label, 0], X_set[y_set == label, 1], color=["red", "green"][i], label=label)
    axes[1].set_title("Original Features")
    axes[1].legend()
    plt.show()


def run_my_own_linear_regression():
    dataset = pd.read_csv("Salary_Data.csv")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

    model = MyOwnLinearRegression(learning_rate=0.01, n_iters=1000, penalty='l1', lambda_param=100)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) 
    
    model.fit(X_train_scaled, y_train, method='batch')

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(scaler.transform(X_line))

    ax = axes[0]
    ax.scatter(X_train, y_train, color="red")
    ax.plot(X_line, y_line, color="blue")
    ax.set_title("Salary vs Experience (Training set - L1)")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")

    ax = axes[1]
    ax.scatter(X_test, y_test, color="red")
    ax.plot(X_line, y_line, color="blue")
    ax.set_title("Salary vs Experience (Test set - L1)")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        run_my_own_linear_regression()
        run_my_own_logistic_regression()
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure dataset files are in the local directory.")