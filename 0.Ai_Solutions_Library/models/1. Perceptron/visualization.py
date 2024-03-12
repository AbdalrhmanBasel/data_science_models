from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    """
    Plots decision regions for a given classifier on a 2D feature space.

    Parameters:
    - X: Features
    - y: Labels
    - classifier: Trained classifier
    - test_idx: Indices of test examples
    - resolution: Grid resolution for decision surface plotting
    """
    markers = ('o', 's', '^', 'v', '<')
    colors = ["red", 'blue', 'lightgreen', 'grey', 'cyan']
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot decision surface
    plot_decision_surface(X, classifier, resolution, cmap)

    # Plot class examples
    plot_class_examples(X, y, markers, colors)

    # Highlight test examples, if provided
    if test_idx is not None and len(test_idx) > 0:
        plot_test_examples(X, y, test_idx)

def plot_decision_surface(X, classifier, resolution, cmap):
    """
    Plots the decision surface of a classifier on a 2D feature space.

    Parameters:
    - X: Features
    - classifier: Trained classifier
    - resolution: Grid resolution for decision surface plotting
    - cmap: Color map for decision surface
    """
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier._predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)

    # Plot decision surface
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

def plot_class_examples(X, y, markers, colors):
    """
    Plots class examples on a 2D feature space.

    Parameters:
    - X: Features
    - y: Labels
    - markers: Marker styles for different classes
    - colors: Colors for different classes
    """
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=f'Class {cl}',
            edgecolor='black'
        )

def plot_test_examples(X, y, test_idx):
    """
    Highlights test examples on a 2D feature space.

    Parameters:
    - X: Features
    - y: Labels
    - test_idx: Indices of test examples
    """
    # Plot all examples
    X_test, y_test = X[test_idx, :], y[test_idx]
    plt.scatter(
        X_test[:, 0], X_test[:, 1],
        c='none',
        edgecolor='black',
        alpha=1.0,
        linewidth=1,
        marker='o',
        s=100,
        label='Test set'
    )


