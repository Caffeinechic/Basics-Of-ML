"""
K-Nearest Neighbors Implementation using Scikit-learn
Production-ready implementation with optimizations
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import cross_val_score
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
    
    # Feature scaling - critical for KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def find_optimal_k(X_train, y_train):
    """
    Find optimal K using cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Optimal K value
    """
    k_range = range(1, 21, 2)  # Odd values to avoid ties
    cv_scores = []
    
    print("\n[Cross-Validation for Optimal K]")
    print("-" * 60)
    print(f"{'K':<5} {'Mean CV Score':<15} {'Std CV Score':<15}")
    print("-" * 60)
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
        print(f"{k:<5} {scores.mean():<15.4f} {scores.std():<15.4f}")
    
    optimal_k = k_range[np.argmax(cv_scores)]
    return optimal_k


def train_model(X_train, y_train, n_neighbors=5, weights='distance', algorithm='auto'):
    """
    Train KNN model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_neighbors: Number of neighbors
        weights: 'uniform' or 'distance'
        algorithm: 'auto', 'ball_tree', 'kd_tree', or 'brute'
    
    Returns:
        Trained model
    """
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        metric='euclidean',
        p=2
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
        Dictionary containing evaluation metrics and predictions
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
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
    
    return metrics, y_test_pred


def display_results(model, metrics, y_test, y_test_pred, X_test):
    """
    Display comprehensive results.
    
    Args:
        model: Trained model
        metrics: Dictionary of evaluation metrics
        y_test: True test values
        y_test_pred: Predicted test values
        X_test: Test features
    """
    print("=" * 70)
    print("K-Nearest Neighbors with Scikit-learn - Results")
    print("=" * 70)
    
    # Model parameters
    print("\n[1] Model Configuration")
    print("-" * 70)
    print(f"Number of neighbors (K): {model.n_neighbors}")
    print(f"Weights: {model.weights}")
    print(f"Distance metric: {model.metric}")
    print(f"Algorithm: {model.algorithm}")
    
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
        status = "Potential overfitting - consider increasing K"
    print(f"Accuracy difference: {acc_diff:.4f} - {status}")
    
    # Sample predictions with probabilities
    print("\n[5] Sample Predictions with Confidence (First 5 test samples)")
    print("-" * 70)
    y_test_proba = model.predict_proba(X_test)
    print(f"{'Actual':<10} {'Predicted':<12} {'Confidence':<12} {'Correct':<10}")
    print("-" * 70)
    for i in range(min(5, len(y_test))):
        confidence = np.max(y_test_proba[i])
        correct = "Yes" if y_test[i] == y_test_pred[i] else "No"
        print(f"{y_test[i]:<10} {y_test_pred[i]:<12} {confidence:<12.4f} {correct:<10}")
    
    # Neighbor analysis for a sample point
    print("\n[6] Neighbor Analysis (First test sample)")
    print("-" * 70)
    distances, indices = model.kneighbors([X_test[0]], return_distance=True)
    print(f"Neighbors for test sample 0:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"  Neighbor {i + 1}: Distance = {dist:.4f}")
    
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
        distances: Distances to neighbors
    """
    new_data = np.array(new_data).reshape(-1, model.n_features_in_)
    new_data_scaled = scaler.transform(new_data)
    predictions = model.predict(new_data_scaled)
    probabilities = model.predict_proba(new_data_scaled)
    distances, indices = model.kneighbors(new_data_scaled)
    return predictions, probabilities, distances


def main():
    """
    Main execution function.
    """
    # File path
    filepath = "../../Datasets/knn_data.csv"
    
    print("\n[1] Loading and preparing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data(filepath)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    print("\n[2] Finding optimal K...")
    optimal_k = find_optimal_k(X_train, y_train)
    print(f"\nOptimal K: {optimal_k}")
    
    print(f"\n[3] Training model with K={optimal_k}...")
    model = train_model(X_train, y_train, n_neighbors=optimal_k, weights='distance')
    print("Model training complete!")
    
    print("\n[4] Evaluating model...")
    metrics, y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    print("\n[5] Displaying results...")
    display_results(model, metrics, y_test, y_test_pred, X_test)
    
    # Example: Making predictions on new data
    print("\n[7] Example: Predicting on new data")
    print("-" * 70)
    new_data = X_test[:3]
    predictions, probabilities, distances = make_prediction(model, scaler, new_data)
    print(f"Predictions: {predictions}")
    print(f"Probabilities:\n{probabilities}")
    print(f"Distances to neighbors:\n{distances}")
    
    print("\nProcess complete!")


if __name__ == "__main__":
    main()
