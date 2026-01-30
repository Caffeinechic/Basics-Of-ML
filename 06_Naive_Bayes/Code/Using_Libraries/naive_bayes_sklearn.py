"""
Naive Bayes Implementation using Scikit-learn
Production-ready implementation with multiple variants
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler
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


def train_model(X_train, y_train, variant='gaussian'):
    """
    Train Naive Bayes model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        variant: Type of Naive Bayes ('gaussian', 'multinomial', 'bernoulli')
    
    Returns:
        Trained model
    """
    if variant == 'gaussian':
        model = GaussianNB()
    elif variant == 'multinomial':
        # Ensure non-negative features
        X_train = np.abs(X_train)
        model = MultinomialNB(alpha=1.0)
    elif variant == 'bernoulli':
        model = BernoulliNB(alpha=1.0)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
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
    y_test_proba = model.predict_proba(X_test)
    
    # Get unique labels
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
    
    return metrics, y_test_pred, y_test_proba


def display_results(model, metrics, y_test, y_test_pred, y_test_proba):
    """
    Display comprehensive results.
    
    Args:
        model: Trained model
        metrics: Dictionary of evaluation metrics
        y_test: True test values
        y_test_pred: Predicted test values
        y_test_proba: Predicted probabilities
    """
    print("=" * 70)
    print("Naive Bayes with Scikit-learn - Results")
    print("=" * 70)
    
    # Model info
    print("\n[1] Model Information")
    print("-" * 70)
    print(f"Model type: {type(model).__name__}")
    print(f"Number of classes: {len(model.classes_)}")
    print(f"Classes: {model.classes_}")
    
    # Class priors
    print("\n[2] Class Prior Probabilities")
    print("-" * 70)
    for i, cls in enumerate(model.classes_):
        prior = np.exp(model.class_log_prior_[i])
        print(f"  Class {cls}: {prior:.4f}")
    
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
    classes = model.classes_
    
    print("       ", end="")
    for c in classes:
        print(f"Pred {c:<5}", end=" ")
    print()
    
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
        status = "Good generalization"
    elif acc_diff < 0.1:
        status = "Acceptable generalization"
    else:
        status = "Potential overfitting"
    print(f"Accuracy difference: {acc_diff:.4f} - {status}")
    
    # Sample predictions
    print("\n[6] Sample Predictions (First 5 test samples)")
    print("-" * 70)
    print(f"{'Actual':<10} {'Predicted':<12} {'Confidence':<15} {'Correct':<10}")
    print("-" * 70)
    for i in range(min(5, len(y_test))):
        confidence = np.max(y_test_proba[i])
        correct = "Yes" if y_test[i] == y_test_pred[i] else "No"
        print(f"{y_test[i]:<10} {y_test_pred[i]:<12} {confidence:<15.4f} {correct:<10}")
    
    print("\n" + "=" * 70)


def make_prediction(model, new_data):
    """
    Make predictions on new data.
    
    Args:
        model: Trained model
        new_data: New data points
    
    Returns:
        predictions: Class labels
        probabilities: Probability scores
    """
    new_data = np.array(new_data).reshape(-1, len(model.theta_[0]))
    predictions = model.predict(new_data)
    probabilities = model.predict_proba(new_data)
    return predictions, probabilities


def main():
    """
    Main execution function.
    """
    # File path
    filepath = "../../Datasets/naive_bayes_data.csv"
    
    print("\n[1] Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_and_prepare_data(filepath)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    print("\n[2] Training Gaussian Naive Bayes model...")
    model = train_model(X_train, y_train, variant='gaussian')
    print("Model training complete!")
    
    print("\n[3] Evaluating model...")
    metrics, y_test_pred, y_test_proba = evaluate_model(
        model, X_train, X_test, y_train, y_test
    )
    
    print("\n[4] Displaying results...")
    display_results(model, metrics, y_test, y_test_pred, y_test_proba)
    
    # Example: Making predictions on new data
    print("\n[7] Example: Predicting on new data")
    print("-" * 70)
    new_data = X_test[:3]
    predictions, probabilities = make_prediction(model, new_data)
    print(f"Predictions: {predictions}")
    print(f"Probabilities:\n{probabilities}")
    
    print("\nProcess complete!")


if __name__ == "__main__":
    main()
