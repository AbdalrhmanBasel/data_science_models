import numpy as np

class LogisticRegression:
    def __init__(self, n_iterations=50, learning_rate=0.01, random_state=1, threshold=0.5):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.threshold = threshold

    def fit(self, X, y):
        r_gen = np.random.RandomState(self.random_state)
        self.w_ = r_gen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0.0
        self.losses_ = []

        for i in range(self.n_iterations):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.learning_rate * X.T.dot(errors) / X.shape[0]
            self.b_ += self.learning_rate * errors.mean()
            loss = -y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output)) / X.shape[0]
            self.losses_.append(loss)

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= self.threshold, 1, 0)