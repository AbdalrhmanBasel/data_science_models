import numpy as np

class Perceptron:
    def __init__(self, n_iterations=50, learning_rate=0.01, random_state=1, threshold=0.5):
        """
        Initialize Perceptron with hyperparameters.

        Parameters:
        - n_iterations: Number of training iterations
        - learning_rate: Learning rate for weight updates
        - random_state: Seed for random number generation
        - threshold: Threshold for making binary predictions
        """
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.threshold = threshold
        self.weights = None  # Initialize weights during fitting
        self.bias = None  # Initialize bias during fitting
        self.losses = []

    def fit(self, X, y):
        """
        Train the Perceptron on the given dataset.

        Parameters:
        - X: Input features
        - y: Target labels
        """
        # Initialize weights and bias
        random_generator = np.random.RandomState(self.random_state)
        self.weights = random_generator.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.bias = 0.0
        self.losses = []

        # Training loop
        for _ in range(self.n_iterations):
            errors = 0
            for xi, target in zip(X, y):
                loss = self._update_weights(xi, target)
                self.losses.append(loss)
                errors += int(loss != 0.0)

    def _update_weights(self, xi, target):
        """
        Update weights and bias based on the given input and target.

        Parameters:
        - xi: Input feature vector
        - target: Target label

        Returns:
        - Loss value
        """
        # Weight update formula: w = w + learning_rate * (target - prediction) * xi
        update = self.learning_rate * (target - self._predict(xi))
        self.weights += update * xi
        # Bias update formula: b = b + learning_rate * (target - prediction)
        self.bias += update
        return int(update != 0.0)

    def _predict(self, X):
        """
        Make predictions based on the given input.

        Parameters:
        - X: Input features

        Returns:
        - Predicted labels (0 or 1)
        """
        # Prediction formula: 1 if net_input >= threshold, else 0
        return np.where(self._net_input(X) >= self.threshold, 1, 0)

    def _net_input(self, X):
        """
        Calculate the net input (weighted sum) for the given input.

        Parameters:
        - X: Input features

        Returns:
        - Net input
        """
        # Net input formula: dot product of weights and input + bias
        return np.dot(X, self.weights) + self.bias
