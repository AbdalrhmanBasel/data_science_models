import numpy as np


class Adaline:
    def __init__(self,n_iterations=50, learning_rate=0.01, threshold=0.5, random_state=1):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.random_state = random_state
        

    def train(self, X,y):
        # init w,b,erros
        rand_no_generator = np.random.RandomState(self.random_state)
        self.weights_ = rand_no_generator.normal(loc=0.0,scale=0.01,size=X.shape[1])
        self.bias_ = 0.0
        self.losses_ = []

        # Train using gradient decent
        for i in range(self.n_iterations):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.weights_ += -self.learning_rate * 2.0 * X.T.dot(errors) / X.shape[0]
            self.bias_ += -self.learning_rate * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        


    def net_input(self,X):
        return np.dot(X, self.weights_) + self.bias_

    def activation(self,X):
        return X

    def predict(self,X):
        return np.where(self.activation(X)> self.threshold,1,0)