import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from models.Adaline_3 import Adaline

# Placeholder data, replace with your actual data
X_std = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 1, 0, 1])

# Instantiate Adaline and train on your data
ada_gd = Adaline(n_iterations=100, learning_rate=0.01)
ada_gd.train(X_std, y)

# Plot decision regions and loss curve
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('o', 's')
    colors = ('red', 'blue')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

if __name__ == "__main__":
    # Plot decision regions
    plot_decision_regions(X_std, y, classifier=ada_gd)
    plt.title('Adaline - Gradient descent')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    # Plot loss curve
    plt.plot(range(1, len(ada_gd.losses_) + 1), ada_gd.losses_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Mean squared error')
    plt.tight_layout()
    plt.show()
