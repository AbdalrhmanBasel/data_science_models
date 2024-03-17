import numpy as np

class AdalinePerceptron:
    def __init__(self, n_iterations=50, learning_rate=0.01, random_state=1, threshold=0.5):
        """
        Initialize Adaline Perceptron with hyperparameters.

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

    def fit(self, X, y):
        """
        Train the Adaline Perceptron model.

        Parameters:
        - X: Input features
        - y: Target labels
        """
        # Initialize weights
        r_gen = np.random.RandomState(self.random_state)
        self.w_ = r_gen.normal(loc=0.0, scale=0.02, size=X.shape[1])
        self.b_ = 0.0
        self.losses_ = []

        # Train model
        for i in range(self.n_iterations):
            loss = self._update_weights(X, y)
            self.losses_.append(loss)

    def _update_weights(self, X, y):
        """
        Update weights and bias based on the given input and target.

        Parameters:
        - X: Input feature matrix
        - y: Target labels

        Returns:
        - loss: Mean squared error loss
        """
        net_input = self._net_input(X)
        output = self.activation(net_input)
        errors = (y - output)
        self.w_ += self.learning_rate * 2.0 * X.T.dot(errors) / X.shape[0]
        self.b_ += self.learning_rate * errors.mean()
        loss = (errors**2).mean()
        return loss

    def _net_input(self, X):
        """
        Calculate the net input (weighted sum) for the given input.

        Parameters:
        - X: Input features

        Returns:
        - Net input
        """
        return np.dot(X, self.w_) + self.b_

    def activation(self, x):
        """
        Linear activation function.

        Parameters:
        - x: Input

        Returns:
        - x
        """
        return x

    def predict(self, X):
        """
        Make binary predictions based on the given input.

        Parameters:
        - X: Input features

        Returns:
        - Binary predictions (0 or 1)
        """
        return np.where(self.activation(self._net_input(X)) >= self.threshold, 1, 0)
