"""
Logistic Regression Implementation using Scikit-learn
Production-ready implementation with best practices
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(filepath):
    """
    Load and prepare data for training.
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
        scaler: Fitted StandardScaler object
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
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train, penalty='l2', C=1.0):
    """
    Train Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        penalty: Regularization type ('l1', 'l2', 'elasticnet', 'none')
        C: Inverse of regularization strength (smaller = stronger)
    
    Returns:
        Trained model
    """
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance comprehensively.
    
    Args:
        model: Trained model
        X_train, X_test: Feature sets
        y_train, y_test: Target sets
    
    Returns:
        Dictionary containing evaluation metrics and predictions
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_auc_roc': roc_auc_score(y_test, y_test_proba),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred)
    }
    
    return metrics, y_test_pred, y_test_proba


def display_results(model, metrics, y_test, y_test_pred, y_test_proba, feature_names=None):
    """
    Display comprehensive results.
    
    Args:
        model: Trained model
        metrics: Dictionary of evaluation metrics
        y_test: True test values
        y_test_pred: Predicted test labels
        y_test_proba: Predicted test probabilities
        feature_names: List of feature names (optional)
    """
    print("=" * 70)
    print("Logistic Regression with Scikit-learn - Results")
    print("=" * 70)
    
    # Model parameters
    print("\n[1] Model Parameters")
    print("-" * 70)
    print(f"Intercept (bias): {model.intercept_[0]:.4f}")
    print("\nCoefficients (weights):")
    if feature_names:
        for name, coef in zip(feature_names, model.coef_[0]):
            print(f"  {name}: {coef:.4f}")
    else:
        for i, coef in enumerate(model.coef_[0]):
            print(f"  Feature {i + 1}: {coef:.4f}")
    
    # Performance metrics
    print("\n[2] Performance Metrics")
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
    print(f"  AUC-ROC: {metrics['test_auc_roc']:.4f}")
    
    # Confusion matrix
    print("\n[3] Confusion Matrix (Test Set)")
    print("-" * 70)
    cm = metrics['confusion_matrix']
    print(f"                Predicted 0    Predicted 1")
    print(f"Actual 0        {cm[0, 0]:<14} {cm[0, 1]:<14}")
    print(f"Actual 1        {cm[1, 0]:<14} {cm[1, 1]:<14}")
    print(f"\nTrue Negatives: {cm[0, 0]}, False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}, True Positives: {cm[1, 1]}")
    
    # Model analysis
    print("\n[4] Model Analysis")
    print("-" * 70)
    acc_diff = metrics['train_accuracy'] - metrics['test_accuracy']
    if acc_diff < 0.05:
        status = "Good fit - minimal overfitting"
    elif acc_diff < 0.1:
        status = "Acceptable fit - slight overfitting"
    else:
        status = "Potential overfitting detected"
    print(f"Accuracy difference: {acc_diff:.4f} - {status}")
    
    # Interpretation
    print("\n[5] Model Interpretation")
    print("-" * 70)
    if metrics['test_precision'] > metrics['test_recall']:
        print("Model is conservative - fewer false positives, more false negatives")
    else:
        print("Model is aggressive - fewer false negatives, more false positives")
    
    # Sample predictions
    print("\n[6] Sample Predictions (First 5 test samples)")
    print("-" * 70)
    print(f"{'Actual':<10} {'Predicted':<12} {'Probability':<15} {'Confidence':<12}")
    print("-" * 70)
    for i in range(min(5, len(y_test))):
        prob = y_test_proba[i]
        confidence = "High" if (prob > 0.8 or prob < 0.2) else "Medium" if (prob > 0.6 or prob < 0.4) else "Low"
        print(f"{y_test[i]:<10} {y_test_pred[i]:<12} {prob:<15.4f} {confidence:<12}")
    
    print("\n" + "=" * 70)


def make_prediction(model, scaler, new_data):
    """
    Make predictions on new data.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        new_data: New data points (2D array or list)
    
    Returns:
        predictions: Class labels
        probabilities: Probability scores
    """
    new_data = np.array(new_data).reshape(-1, model.n_features_in_)
    new_data_scaled = scaler.transform(new_data)
    predictions = model.predict(new_data_scaled)
    probabilities = model.predict_proba(new_data_scaled)[:, 1]
    return predictions, probabilities


def main():
    """
    Main execution function.
    """
    # File path
    filepath = "../../Datasets/classification_data.csv"
    
    print("\n[1] Loading and preparing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data(filepath)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Class 0 count (train): {np.sum(y_train == 0)}")
    print(f"Class 1 count (train): {np.sum(y_train == 1)}")
    
    print("\n[2] Training model...")
    model = train_model(X_train, y_train, penalty='l2', C=1.0)
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
    new_data = X_test[:3]  # Using first 3 test samples as example
    predictions, probabilities = make_prediction(model, scaler, new_data)
    print(f"Predictions: {predictions}")
    print(f"Probabilities: {probabilities}")
    
    print("\nProcess complete!")


if __name__ == "__main__":
    main()
