"""
Decision Tree Implementation using Scikit-learn
Production-ready implementation with visualization
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(filepath):
    """
    Load and prepare data for training.
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    # Load data
    data = pd.read_csv(filepath)
    
    # Separate features and target
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Train-test split (80-20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, max_depth=5, min_samples_split=2, 
                min_samples_leaf=1, criterion='gini'):
    """
    Train Decision Tree model.
    
    Args:
        X_train: Training features
        y_train: Training target
        max_depth: Maximum depth of tree
        min_samples_split: Minimum samples to split node
        min_samples_leaf: Minimum samples in leaf node
        criterion: Split criterion ('gini' or 'entropy')
    
    Returns:
        Trained model
    """
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_train, X_test: Feature sets
        y_train, y_test: Target sets
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Get unique labels for handling multiclass
    labels = np.unique(y_test)
    average_method = 'binary' if len(labels) == 2 else 'weighted'
    
    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_precision': precision_score(y_train, y_train_pred, average=average_method, zero_division=0),
        'test_precision': precision_score(y_test, y_test_pred, average=average_method, zero_division=0),
        'train_recall': recall_score(y_train, y_train_pred, average=average_method, zero_division=0),
        'test_recall': recall_score(y_test, y_test_pred, average=average_method, zero_division=0),
        'train_f1': f1_score(y_train, y_train_pred, average=average_method, zero_division=0),
        'test_f1': f1_score(y_test, y_test_pred, average=average_method, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred)
    }
    
    return metrics, y_test_pred


def display_results(model, metrics, y_test, y_test_pred, feature_names=None):
    """
    Display comprehensive results.
    
    Args:
        model: Trained model
        metrics: Dictionary of evaluation metrics
        y_test: True test values
        y_test_pred: Predicted test values
        feature_names: List of feature names (optional)
    """
    print("=" * 70)
    print("Decision Tree Classifier with Scikit-learn - Results")
    print("=" * 70)
    
    # Tree structure info
    print("\n[1] Tree Structure Information")
    print("-" * 70)
    print(f"Number of nodes: {model.tree_.node_count}")
    print(f"Number of leaves: {model.get_n_leaves()}")
    print(f"Maximum depth: {model.get_depth()}")
    print(f"Number of features: {model.n_features_in_}")
    
    # Feature importance
    print("\n[2] Feature Importance")
    print("-" * 70)
    if feature_names:
        for name, importance in zip(feature_names, model.feature_importances_):
            if importance > 0:
                print(f"  {name}: {importance:.4f}")
    else:
        for i, importance in enumerate(model.feature_importances_):
            if importance > 0:
                print(f"  Feature {i + 1}: {importance:.4f}")
    
    # Performance metrics
    print("\n[3] Performance Metrics")
    print("-" * 70)
    print("Training Set:")
    print(f"  Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Precision: {metrics['train_precision']:.4f}")
    print(f"  Recall: {metrics['train_recall']:.4f}")
    print(f"  F1 Score: {metrics['train_f1']:.4f}")
    
    print("\nTest Set:")
    print(f"  Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  Precision: {metrics['test_precision']:.4f}")
    print(f"  Recall: {metrics['test_recall']:.4f}")
    print(f"  F1 Score: {metrics['test_f1']:.4f}")
    
    # Confusion matrix
    print("\n[4] Confusion Matrix (Test Set)")
    print("-" * 70)
    cm = metrics['confusion_matrix']
    classes = np.unique(y_test)
    
    # Header
    print("       ", end="")
    for c in classes:
        print(f"Pred {c:<5}", end=" ")
    print()
    
    # Rows
    for i, c in enumerate(classes):
        print(f"Act {c:<3}", end=" ")
        for j in range(len(classes)):
            print(f"{cm[i, j]:<10}", end=" ")
        print()
    
    # Model analysis
    print("\n[5] Model Analysis")
    print("-" * 70)
    acc_diff = metrics['train_accuracy'] - metrics['test_accuracy']
    if acc_diff < 0.05:
        status = "Good fit - minimal overfitting"
    elif acc_diff < 0.1:
        status = "Acceptable fit - slight overfitting"
    else:
        status = "Overfitting detected - consider pruning"
    print(f"Accuracy difference: {acc_diff:.4f} - {status}")
    
    # Sample predictions
    print("\n[6] Sample Predictions (First 10 test samples)")
    print("-" * 70)
    print(f"{'Actual':<10} {'Predicted':<12} {'Correct':<10}")
    print("-" * 70)
    for i in range(min(10, len(y_test))):
        correct = "Yes" if y_test[i] == y_test_pred[i] else "No"
        print(f"{y_test[i]:<10} {y_test_pred[i]:<12} {correct:<10}")
    
    print("\n" + "=" * 70)


def make_prediction(model, new_data):
    """
    Make predictions on new data.
    
    Args:
        model: Trained model
        new_data: New data points (2D array or list)
    
    Returns:
        predictions: Class labels
        probabilities: Probability scores for each class
    """
    new_data = np.array(new_data).reshape(-1, model.n_features_in_)
    predictions = model.predict(new_data)
    probabilities = model.predict_proba(new_data)
    return predictions, probabilities


def main():
    """
    Main execution function.
    """
    # File path
    filepath = "../../Datasets/tree_data.csv"
    
    print("\n[1] Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_and_prepare_data(filepath)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    print("\n[2] Training model...")
    model = train_model(X_train, y_train, max_depth=5, criterion='gini')
    print("Model training complete!")
    
    print("\n[3] Evaluating model...")
    metrics, y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    print("\n[4] Displaying results...")
    display_results(model, metrics, y_test, y_test_pred)
    
    # Example: Making predictions on new data
    print("\n[7] Example: Predicting on new data")
    print("-" * 70)
    new_data = X_test[:3]  # Using first 3 test samples as example
    predictions, probabilities = make_prediction(model, new_data)
    print(f"Predictions: {predictions}")
    print(f"Probabilities:\n{probabilities}")
    
    print("\nProcess complete!")


if __name__ == "__main__":
    main()
