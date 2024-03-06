import numpy as np

class Adaline:
    def __init__(self, n_iterations=50, learning_rate=0.01, random_state=1, threshold=0.5):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.w = None
        self.b = 0.0
        self.losses = []  # Corrected variable name: 'loses' to 'losses'
        self.threshold = threshold

    def train(self, X, y):
        # Initialize weights and bias
        r_gen = np.random.RandomState(self.random_state)
        self.w = r_gen.normal(loc=0, scale=0.01, size=X.shape[1])
        self.b = 0.0
        self.losses = []  # Reset losses for each training iteration

        for _ in range(self.n_iterations):
            for x, target in zip(X, y):
                # net input
                net_input = self.net_input(x)
                # activation
                output = self.activation(net_input)
                # calculate errors
                errors = target - output
                # update weights using MSE
                self.w += self.learning_rate * 2.0 * x.T.dot(errors) / X.shape[0]
                # update bias using MSE
                self.b += self.learning_rate * 2.0 * errors.mean()
                # calculate the loss **2 to simplify equation and get mean
                loss = (errors ** 2).mean()
                # append the loss for later usage
                self.losses.append(loss)
        return self

    def net_input(self, X):
        return np.dot(self.w, X) + self.b

    def activation(self, x):
        return x

    def predict(self, X):
        return np.where(self.activation(X) > self.threshold, 1, 0)
