from Perceptron import Perceptron
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    # URL
    DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    df = pd.read_csv(DATASET_URL,header=None,encoding="utf-8")
    # print(df.tail())

    y = df.iloc[0:100,4].values
    y = np.where(y == "Iris-setosa",0,1)

    X = df.iloc[0:100,[0,2]].values

    # plt.scatter(X[:50,0], X[:50,1], color="red", market="o", label="Setosa")
    # plt.scatter(X[50:100,0], X[50:100,1], color="blue", market="s", label="Versicolor")
    # plt.xlabel("Sepal length [cm]")
    # plt.ylabel("Petal length [cm]")
    # plt.legend(loc='upper left')
    # plt.show()


    model = Perceptron(0.1,10)
    model.fit(X,y)

    plt.plot(range(1, len(model.errors_) + 1), model.errors_, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel("Number of updates")
    plt.show
    