import numpy as np


class Perceptron:
    def __init__(self, n_iterations=50, learning_rate = 0.01, random_state=1):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.random_state = random_state


    def train(self, X,y):
        # init w,b, and error
        random_generator = np.random.RandomState(self.random_state)
        self.weight_ = random_generator.normal(loc=0.0,scale=0.01, size=X.shape[1]) # mean = 0.0, std = 0.01, size=X[1]
        self.bias_ = 0.0
        self.errors = [] 

        # training
        for i in range(self.n_iterations):
            errors = 0
            for x,target in zip(X,y):
                update = self.learning_rate * (target - self.predict(x))
                self.weight_ += update * x
                self.bias_ += update * x
                self.errors += int(update != 0.0) # if there is error space, add it to erros
            self.errors_.append(errors)


    def net_input(self,X):
        return np.dot(self.weight_, X) + self.bias_

    def predict(self,X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)