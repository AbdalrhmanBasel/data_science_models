from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from model import DecisionTree
import numpy as np
import matplotlib.pyplot as plt
from visualization import plot_decision_regions 

if __name__ == "__main__":
    # 1. Preprocessing
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]] 
    y = iris.target
    target_names = iris.target_names 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # 2. Training
    model = DecisionTree(criterion="gini", max_depth=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 3. Evaluation
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)

    # 4. Visualization
    # X_combined = np.vstack((X_train, X_test))
    # y_combined = np.hstack((y_train, y_test))

    # plot_decision_regions(X_train, y_train, classifier=model, test_idx=range(105, 150))
    # plt.xlabel("Petal Length")
    # plt.ylabel("Petal Width")
    # plt.show()
