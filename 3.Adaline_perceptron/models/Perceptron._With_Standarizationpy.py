import numpy as np

class Adaline_Perceptron:
    def __init__(self, n_iterations=50, learning_rate=0.01, random_state=1, threshold=0.5):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.threshold = threshold
        self.w = None
        self.b = None
        self.losses = []

    def train(self, X, y):
        # Standardize features
        X_std = self.standarization(X)

        # Initialization
        r_gen = np.random.RandomState(self.random_state)
        self.w = r_gen.normal(loc=0, scale=0.001, size=X_std.shape[1])
        self.b = 0.0
        self.losses = []

        for i in range(self.n_iterations):
            net_input = self.net_input(X_std)
            output = self.activation(net_input)
            errors = y - output

            # Update weights and bias
            self.w += self.learning_rate * 2.0 * X_std.T.dot(errors) / X_std.shape[0]
            self.b += self.learning_rate * 2.0 * errors.mean()

            # Calculate the loss
            loss = (errors ** 2).mean()
            self.losses.append(loss)

        return self

    def standarization(self, X):
        X_std = np.copy(X)
        X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
        X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
        return X_std

    def net_input(self, X):
        return np.dot(X, self.w) + self.b

    def activation(self, x):
        return x

    def predict(self, X):
        return np.where(self.activation(X) > self.threshold, 1, 0)
