

"""
THIS MODEL IS NOT OPTIMIZED - IN PROGRESS!
"""

import numpy as np

class SupportVectorMachine:
    def __init__(self, n_iterations=50, learning_rate=0.01, random_state=1, threshold=0.5, lambda_param=0.01):
        """
        Initialize Support Vector Machine parameters.
        
        Parameters:
        - n_iterations (int): Number of iterations for training.
        - learning_rate (float): Learning rate for updating weights.
        - random_state (int): Random seed for reproducibility.
        - threshold (float): Threshold for classification.
        - lambda_param (float): Regularization parameter.
        """
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.threshold = threshold
        self.lambda_param = lambda_param

    def train(self, X, y):
        """
        Train the Support Vector Machine model.
        
        Parameters:
        - X (array-like): Input data of shape (n_samples, n_features).
        - y (array-like): Target values of shape (n_samples,).
        """
        r_gen = np.random.RandomState(self.random_state)
        self.w_ = r_gen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0.0
        self.losses_ = []

        for _ in range(self.n_iterations):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = self.calculate_hinge_loss(X, y, output)
            gradient = np.where(errors[:, np.newaxis] > 0, -X * errors[:, np.newaxis], 0)
            self.w_ -= self.learning_rate * (np.mean(gradient, axis=0) + 2 * self.lambda_param * self.w_)
            self.b_ += self.learning_rate * np.mean(errors)
            loss = np.mean(np.maximum(0, 1 - y * net_input)) + self.lambda_param * np.dot(self.w_, self.w_)
            self.losses_.append(loss)

    def activation(self, z):
        """
        Identity function as activation.
        
        Parameters:
        - z (array-like): Input data.
        
        Returns:
        - z (array-like): Identity activation.
        """
        return z
    
    def net_input(self, X):
        """
        Compute the net input.
        
        Parameters:
        - X (array-like): Input data.
        
        Returns:
        - net_input (array-like): Dot product of weights and input data plus bias.
        """
        return np.dot(X, self.w_) + self.b_

    def calculate_hinge_loss(self, X, y, output):
        """
        Calculate hinge loss.
        
        Parameters:
        - X (array-like): Input data.
        - y (array-like): Target values.
        - output (array-like): Predicted output.
        
        Returns:
        - errors (array-like): Errors based on hinge loss.
        """
        errors = np.maximum(0, 1 - y * output)
        errors = np.where(errors == 0, 0, -y)
        return errors
    
    def predict(self, X):
        """
        Predict the class labels.
        
        Parameters:
        - X (array-like): Input data.
        
        Returns:
        - predictions (array-like): Predicted class labels.
        """
        return np.where(self.activation(self.net_input(X)) >= self.threshold, 1, -1)
 
















from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert target labels to -1 and 1 for SVM
y_train_svm = np.where(y_train == 0, -1, 1)
y_test_svm = np.where(y_test == 0, -1, 1)

# Train the Support Vector Machine model
svm = SupportVectorMachine(n_iterations=1000, learning_rate=0.01, random_state=1, threshold=0.0)
svm.train(X_train_scaled, y_train_svm)

# Predictions
predictions = svm.predict(X_test_scaled)

# Calculate accuracy
accuracy = np.mean(predictions == y_test_svm)
print("Accuracy:", accuracy)
