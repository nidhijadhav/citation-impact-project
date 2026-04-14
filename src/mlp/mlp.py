import numpy as np


class MLP:
    """
    A manual NumPy MLP implementation for binary classification.

    Architecture:
        input -> Linear -> ReLU -> Linear -> Sigmoid
    """

    def __init__(self, input_dim, hidden_dim=32, seed=42):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units in the hidden layer.
            seed (int): Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)

        # ReLU layer init
        self.W1 = rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))

        # output layer init
        self.W2 = rng.standard_normal((hidden_dim, 1)) * np.sqrt(1.0 / hidden_dim)
        self.b2 = np.zeros((1, 1))

        # forward-pass cache
        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None

        # gradient cache
        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, X):
        """
        Performs forward pass and stores intermediate values.
        Args:
            X (np.ndarray): Input features of shape (n_samples, input_dim)
        Returns:
            np.ndarray: Output probabilities of shape (n_samples, 1)
        """
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    def compute_loss(self, y_true, y_pred):
        """
        Computes binary cross-entropy loss.
        Args:
            y_true (np.ndarray): True labels of shape (n_samples, 1)
            y_pred (np.ndarray): Predicted probabilities of shape (n_samples, 1)
        Returns:
            float: Loss value
        """
        y_true = np.asarray(y_true).reshape(-1, 1)
        y_pred = np.asarray(y_pred).reshape(-1, 1)

        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)

        loss = -np.mean(
            y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)
        )
        
        return float(loss)

    def backward(self, X, y_true):
        """
        Performs backward pass to compute gradients.
        Args:
            X (np.ndarray): Input features of shape (n_samples, input_dim)
            y_true (np.ndarray): True labels of shape (n_samples, 1)
        Returns:
            None
        """
        m = X.shape[0]
        y_true = np.asarray(y_true).reshape(-1, 1)

        # output layer gradients
        dZ2 = self.A2 - y_true
        self.dW2 = (self.A1.T @ dZ2) / m
        self.db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # hidden layer gradients
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        self.dW1 = (X.T @ dZ1) / m
        self.db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    def update_params(self, learning_rate=0.01):
        """
        Updates parameters using gradient descent.
        Args:
            learning_rate (float): Learning rate for parameter updates
        Returns:
            None
        """
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

    def train_step(self, X, y_true, learning_rate=0.01):
        """
        Performs a single training step: forward pass, loss computation, backward pass, and parameter update.
        Args:
            X (np.ndarray): Input features of shape (n_samples, input_dim)
            y_true (np.ndarray): True labels of shape (n_samples, 1)
            learning_rate (float): Learning rate for parameter updates
        Returns:
            float: Loss value for the current training step
        """
        y_pred = self.forward(X)
        loss = self.compute_loss(y_true, y_pred)
        self.backward(X, y_true)
        self.update_params(learning_rate)
        return loss

    def predict_proba(self, X):
        """
        Returns predicted probabilities.
        Args:
            X (np.ndarray): Input features of shape (n_samples, input_dim)
        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, 1)
        """
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        """
        Returns binary predictions based on a threshold.
        Args:
            X (np.ndarray): Input features of shape (n_samples, input_dim)
            threshold (float): Threshold for converting probabilities to binary predictions
        Returns:
            np.ndarray: Binary predictions of shape (n_samples, 1)
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int).flatten()