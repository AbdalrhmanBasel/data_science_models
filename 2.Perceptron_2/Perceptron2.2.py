import numpy as np

class Perceptron:
    def __init__ (self, n_iterations=50, learning_rate=0.01, random_state=1,threshold = 0.5):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.weights = None
        self.biases = None
        self.output = []
        self.threshold = threshold

    def train(self, X,y):
        r_gen = np.random.RandomState(self.random_state)
        self.weights = r_gen.normal(loc=0, scale=0.01, size=X.shape[1])
        self.biases = 0.0
        self.errors_ = []

        # Train
        for i in range(self.n_iterations):
            errors = 0
            for x,target in zip(X,y):
                update = self.learning_rate * (target - self.predict(x))
                self.weights += update * x
                self.biases += update 
                errors += int(update != 0.0) 
            self.errors_.append(errors)


    def net_input(self,X):
        z = np.dot(self.weights, X) + self.biases
        return z 

    def predict(self,X):
        return np.where(self.net_input(X) >= self.threshold, 1,0)