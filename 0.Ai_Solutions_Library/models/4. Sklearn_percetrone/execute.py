from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from visualization import plot_decision_regions


def data_visualization(model, X, y, X_test, y_test):
    """
    Visualize the decision regions of the Perceptron model.

    Parameters:
    - model: Trained Perceptron model
    - X: Entire dataset features
    - y: Entire dataset labels
    - X_test: Testing data features
    - y_test: Testing data labels
    """
    plt.figure(figsize=(10, 6))
    plt.title('Perceptron Decision Regions')
    model.fit(X, y)  # Fit on the entire dataset for plotting
    plot_decision_regions(X, y, classifier=model)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', edgecolor='black', label='Test set')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    # Data preprocessing
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    
    print("Class labels:", np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # Data standardization
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Model training
    ppn = Perceptron(eta0=0.1, random_state=1)
    ppn.fit(X_train_std, y_train)

    # Evaluate model
    y_pred = ppn.predict(X_test)
    
    classification_rep = classification_report(y_test, y_pred)

    print("Classification Report:")
    print(classification_rep)

    data_visualization(ppn, X_train_std, y_train, X_test_std, y_test)