"""
Support Vector Machine Implementation using Scikit-learn
Production-ready implementation with kernel support
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import GridSearchCV
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
    
    # Feature scaling - critical for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train, kernel='rbf', C=1.0, gamma='scale'):
    """
    Train SVM model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        C: Regularization parameter
        gamma: Kernel coefficient
    
    Returns:
        Trained model
    """
    # Cast to proper types for sklearn
    from typing import Literal
    kernel_type: Literal['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] = kernel  # type: ignore
    gamma_type: float | Literal['scale', 'auto'] = gamma  # type: ignore
    
    model = SVC(
        kernel=kernel_type,
        C=C,
        gamma=gamma_type,
        random_state=42,
        probability=True  # Enable probability estimates
    )
    model.fit(X_train, y_train)
    return model


def tune_hyperparameters(X_train, y_train):
    """
    Tune hyperparameters using grid search.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Best parameters
    """
    print("\n[Hyperparameter Tuning with Grid Search]")
    print("-" * 70)
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf']
    }
    
    grid_search = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid,
        cv=3,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_


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
    print("Support Vector Machine with Scikit-learn - Results")
    print("=" * 70)
    
    # Model configuration
    print("\n[1] Model Configuration")
    print("-" * 70)
    print(f"Kernel: {model.kernel}")
    print(f"C (Regularization): {model.C}")
    print(f"Gamma: {model.gamma}")
    print(f"Number of support vectors: {len(model.support_vectors_)}")
    print(f"Support vectors per class: {model.n_support_}")
    
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
    
    # Confusion matrix
    print("\n[3] Confusion Matrix (Test Set)")
    print("-" * 70)
    cm = metrics['confusion_matrix']
    classes = np.unique(y_test)
    
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
    print("\n[4] Model Analysis")
    print("-" * 70)
    acc_diff = metrics['train_accuracy'] - metrics['test_accuracy']
    if acc_diff < 0.05:
        status = "Good generalization"
    elif acc_diff < 0.1:
        status = "Acceptable generalization"
    else:
        status = "Potential overfitting - consider adjusting C or gamma"
    print(f"Accuracy difference: {acc_diff:.4f} - {status}")
    
    # Support vector analysis
    sv_percentage = (len(model.support_vectors_) / model.shape_fit_[0]) * 100
    print(f"\nSupport vectors: {len(model.support_vectors_)} ({sv_percentage:.1f}% of training data)")
    if sv_percentage < 30:
        print("  Interpretation: Good margin, well-separated classes")
    elif sv_percentage < 60:
        print("  Interpretation: Moderate margin")
    else:
        print("  Interpretation: Poor margin, consider different kernel or parameters")
    
    # Sample predictions
    print("\n[5] Sample Predictions (First 5 test samples)")
    print("-" * 70)
    print(f"{'Actual':<10} {'Predicted':<12} {'Confidence':<15} {'Correct':<10}")
    print("-" * 70)
    for i in range(min(5, len(y_test))):
        confidence = np.max(y_test_proba[i])
        correct = "Yes" if y_test[i] == y_test_pred[i] else "No"
        print(f"{y_test[i]:<10} {y_test_pred[i]:<12} {confidence:<15.4f} {correct:<10}")
    
    print("\n" + "=" * 70)


def make_prediction(model, scaler, new_data):
    """
    Make predictions on new data.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        new_data: New data points
    
    Returns:
        predictions: Class labels
        probabilities: Probability scores
        decision_function: Decision function values
    """
    new_data = np.array(new_data).reshape(-1, model.n_features_in_)
    new_data_scaled = scaler.transform(new_data)
    predictions = model.predict(new_data_scaled)
    probabilities = model.predict_proba(new_data_scaled)
    decision_function = model.decision_function(new_data_scaled)
    return predictions, probabilities, decision_function


def main():
    """
    Main execution function.
    """
    # File path
    filepath = "../../Datasets/svm_data.csv"
    
    print("\n[1] Loading and preparing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data(filepath)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(np.array(y_train)))}")
    
    # Optional: Tune hyperparameters (commented out for speed)
    # best_params = tune_hyperparameters(X_train, y_train)
    
    print("\n[2] Training SVM model...")
    model = train_model(X_train, y_train, kernel='rbf', C=1.0, gamma='scale')
    print("Model training complete!")
    
    print("\n[3] Evaluating model...")
    metrics, y_test_pred, y_test_proba = evaluate_model(
        model, X_train, X_test, y_train, y_test
    )
    
    print("\n[4] Displaying results...")
    display_results(model, metrics, y_test, y_test_pred, y_test_proba)
    
    # Example: Making predictions on new data
    print("\n[6] Example: Predicting on new data")
    print("-" * 70)
    new_data = X_test[:3]
    predictions, probabilities, decision_vals = make_prediction(model, scaler, new_data)
    print(f"Predictions: {predictions}")
    print(f"Probabilities:\n{probabilities}")
    print(f"Decision function values:\n{decision_vals}")
    
    print("\nProcess complete!")


if __name__ == "__main__":
    main()
