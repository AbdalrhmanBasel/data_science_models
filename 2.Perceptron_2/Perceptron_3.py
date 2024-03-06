import numpy as np

class Perceptron:
    def __init__(self, n_iterations=50, learning_rate=0.01, random_state=1):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.random_state = random_state

    def train(self,X):
        r_gen = np.random.RandomState(self.random_state)
        self.w_ = r_gen.normal(loc=0.0, scale=0.01, size=shape[1])
        self.b_ = 0.0
        self.errors = []

        for _ in range(self.n_iterations):
            errors = 0
            for x,target in zip(X,y):
                update +=  self.learning_rate * (target - self.predict(x))
                self.w_ += update * x
                self.b_ += update
                self.errors_ += int(update != 0.0)
            self.errors_.append(errors)

                
            
    def net_input(self,X):
        return np.dot(self.w_, X) + self.b

    def predict(self,X):
        return np.where(self.net_input(X) >= 0, 1, 0)

