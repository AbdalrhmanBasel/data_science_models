from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from visualization import plot_decision_regions


def standardize_data(X_train, X_test):
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std, X_test_std, sc


def data_preprocessing():
    """
    Load the Iris dataset, split it into training and testing sets, and standardize the features.

    Returns:
    - X_train: Training data features
    - X_test: Testing data features
    - y_train: Training data labels
    - y_test: Testing data labels
    - X_train_std: Standardized training data features
    - X_test_std: Standardized testing data features
    - sc: StandardScaler object for future use
    """
    iris = datasets.load_iris()
    X = iris.data[:100, :2]
    y = iris.target[:100]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train_std, X_test_std, sc = standardize_data(X_train, X_test)
    return X_train, X_test, y_train, y_test, X_train_std, X_test_std, sc


def model_training(X_train_std, y_train):
    model = LogisticRegression(C=100.0, solver='lbfgs',multi_class='ovr')
    model.fit(X_train_std, y_train)
    return model


def model_evaluation(model, X_test, y_test):
    """
    Evaluate the Logistic Regression model and print various evaluation metrics.

    Parameters:
    - perceptron: Trained Perceptron model
    - X_test: Testing data features
    - y_test: Testing data labels
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions)
    classification_rep = classification_report(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(confusion)
    print("Classification Report:")
    print(classification_rep)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


def data_visualization(model, X, y, X_test, y_test):
    """
    Visualize the decision regions of the Perceptron model.

    Parameters:
    - perceptron: Trained Perceptron model
    - X: Entire dataset features
    - y: Entire dataset labels
    - X_test: Testing data features
    - y_test: Testing data labels
    """
    plt.figure(figsize=(10, 6))
    plt.title('Logistic Regression Decision Regions')
    model.fit(X, y)  # Fit on the entire dataset for plotting
    plot_decision_regions(X, y, classifier=model)
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. Data preprocessing
    X_train, X_test, y_train, y_test, X_train_std, X_test_std, sc = data_preprocessing()

    # Model training
    perceptron = model_training(X_train_std, y_train)

    # Model evaluation
    model_evaluation(perceptron, X_test_std, y_test)

    # Data visualization
    data_visualization(perceptron, X_train_std, y_train, X_test_std, y_test)