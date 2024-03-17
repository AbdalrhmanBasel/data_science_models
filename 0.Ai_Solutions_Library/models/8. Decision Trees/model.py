import numpy as np

class DecisionTree:
    def __init__(self, criterion='gini', max_depth=None):
        """
        Initialize DecisionTree instance.

        Parameters:
        - criterion: Splitting criterion, either 'gini' (default) or 'entropy'.
        - max_depth: Maximum depth of the tree. If None, the tree will be fully grown.

        Attributes:
        - criterion: The chosen splitting criterion.
        - max_depth: Maximum depth of the tree.
        """
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X, y):
        """
        Build a decision tree classifier from the training set (X, y).

        Parameters:
        - X: Training data, shape (n_samples, n_features).
        - y: Target values, shape (n_samples,).
        """
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """
        Build the tree recursively.

        Parameters:
        - X: Training data, shape (n_samples, n_features).
        - y: Target values, shape (n_samples,).
        - depth: Current depth of the node in the tree.

        Returns:
        - node: Dictionary representing a node in the decision tree.
        """
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        node = {'predicted_class': predicted_class}

        if depth < self.max_depth:
            best_feature, best_threshold = self._find_best_split(X, y)
            if best_feature is not None:
                indices_left = X[:, best_feature] < best_threshold
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['feature_index'] = best_feature
                node['threshold'] = best_threshold
                node['left'] = self._grow_tree(X_left, y_left, depth + 1)
                node['right'] = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _find_best_split(self, X, y):
        """
        Find the best split for a node.

        Parameters:
        - X: Training data, shape (n_samples, n_features).
        - y: Target values, shape (n_samples,).

        Returns:
        - best_feature: Index of the best feature for splitting.
        - best_threshold: Threshold value for the best split.
        """
        best_impurity = np.inf
        best_feature, best_threshold = None, None

        for feature in range(self.n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                impurity = self._compute_impurity(X, y, feature, threshold)
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _compute_impurity(self, X, y, feature, threshold):
        """
        Compute impurity for a split.

        Parameters:
        - X: Training data, shape (n_samples, n_features).
        - y: Target values, shape (n_samples,).
        - feature: Index of the feature for splitting.
        - threshold: Threshold value for splitting.

        Returns:
        - impurity: Impurity value for the split.
        """
        if self.criterion == 'gini':
            return self._gini_impurity(X, y, feature, threshold)
        elif self.criterion == 'entropy':
            return self._entropy(X, y, feature, threshold)

    def _gini_impurity(self, X, y, feature, threshold):
        """
        Compute Gini impurity for a split.

        Parameters:
        - X: Training data, shape (n_samples, n_features).
        - y: Target values, shape (n_samples,).
        - feature: Index of the feature for splitting.
        - threshold: Threshold value for splitting.

        Returns:
        - gini: Gini impurity value for the split.
        """
        indices_left = X[:, feature] < threshold
        y_left = y[indices_left]
        y_right = y[~indices_left]

        gini_left = 1.0 - sum((np.sum(y_left == c) / len(y_left)) ** 2 for c in range(self.n_classes))
        gini_right = 1.0 - sum((np.sum(y_right == c) / len(y_right)) ** 2 for c in range(self.n_classes))

        gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)
        return gini

    def _entropy(self, X, y, feature, threshold):
        """
        Compute entropy for a split.

        Parameters:
        - X: Training data, shape (n_samples, n_features).
        - y: Target values, shape (n_samples,).
        - feature: Index of the feature for splitting.
        - threshold: Threshold value for splitting.

        Returns:
        - entropy: Entropy value for the split.
        """
        indices_left = X[:, feature] < threshold
        y_left = y[indices_left]
        y_right = y[~indices_left]

        entropy_left = self._calculate_entropy(y_left)
        entropy_right = self._calculate_entropy(y_right)

        entropy = (len(y_left) * entropy_left + len(y_right) * entropy_right) / len(y)
        return entropy

    def _calculate_entropy(self, y):
        """
        Calculate entropy for a set of target values.

        Parameters:
        - y: Target values, shape (n_samples,).

        Returns:
        - entropy: Entropy value for the target values.
        """
        unique_classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def _predict(self, inputs):
        """
        Predict the class label for a single sample.

        Parameters:
        - inputs: Feature values for a single sample.

        Returns:
        - predicted_class: Predicted class label.
        """
        node = self.tree_
        while 'threshold' in node:
            if inputs[node['feature_index']] < node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['predicted_class']

    def predict(self, X):
        """
        Predict the class labels for multiple samples.

        Parameters:
        - X: Feature values for multiple samples.

        Returns:
        - predicted_classes: Predicted class labels for the samples.
        """
        return [self._predict(inputs) for inputs in X]

if __name__ == "__main__":
    # Example usage
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Train decision tree classifier with entropy criterion
    tree_model = DecisionTree(criterion='entropy', max_depth=3)
    tree_model.fit(X_train, y_train)

    # Predict on test set
    y_pred = tree_model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
