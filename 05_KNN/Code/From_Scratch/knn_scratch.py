"""
K-Nearest Neighbors Implementation from Scratch
Using only NumPy - implements actual distance calculations and voting logic
"""

import numpy as np
import pandas as pd
from collections import Counter


class KNNScratch:
    """
    K-Nearest Neighbors implemented from scratch.
    
    Attributes:
        k: Number of neighbors to consider
        distance_metric: Distance metric to use ('euclidean' or 'manhattan')
        weights: Weighting scheme ('uniform' or 'distance')
        X_train: Training features
        y_train: Training labels
    """
    
    def __init__(self, k=5, distance_metric='euclidean', weights='uniform'):
        """
        Initialize KNN classifier.
        
        Args:
            k: Number of neighbors (default: 5)
            distance_metric: 'euclidean' or 'manhattan' (default: 'euclidean')
            weights: 'uniform' or 'distance' (default: 'uniform')
        """
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        Store training data (lazy learning - no actual training).
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def _calculate_distance(self, x1, x2):
        """
        Calculate distance between two points.
        
        Args:
            x1: First point
            x2: Second point
        
        Returns:
            Distance value
        """
        if self.distance_metric == 'euclidean':
            # Euclidean distance: sqrt(sum((x1 - x2)^2))
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            # Manhattan distance: sum(|x1 - x2|)
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _get_neighbors(self, x):
        """
        Find K nearest neighbors for a single point.
        
        Args:
            x: Query point
        
        Returns:
            Indices of K nearest neighbors
        """
        # Calculate distances to all training points
        distances = np.array([self._calculate_distance(x, x_train) 
                             for x_train in self.X_train])
        
        # Get indices of K smallest distances
        k_indices = np.argsort(distances)[:self.k]
        
        return k_indices, distances[k_indices]
    
    def _predict_classification(self, x):
        """
        Predict class for a single point.
        
        Args:
            x: Query point
        
        Returns:
            Predicted class label
        """
        # Get K nearest neighbors
        k_indices, distances = self._get_neighbors(x)
        k_nearest_labels = self.y_train[k_indices]
        
        if self.weights == 'uniform':
            # Majority voting
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]
        
        elif self.weights == 'distance':
            # Distance-weighted voting
            # Avoid division by zero
            weights = 1 / (distances + 1e-10)
            
            # For each unique class, sum the weights
            unique_classes = np.unique(k_nearest_labels)
            weighted_votes = {}
            
            for cls in unique_classes:
                cls_weights = weights[k_nearest_labels == cls]
                weighted_votes[cls] = np.sum(cls_weights)
            
            # Return class with highest weighted vote
            return max(weighted_votes, key=weighted_votes.get)
    
    def predict(self, X):
        """
        Predict classes for multiple points.
        
        Args:
            X: Query points, shape (n_samples, n_features)
        
        Returns:
            Predicted labels, shape (n_samples,)
        """
        X = np.array(X)
        predictions = [self._predict_classification(x) for x in X]
        return np.array(predictions)
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Args:
            X: Features
            y: True labels
        
        Returns:
            Accuracy
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def main():
    """
    Demonstration of KNN from scratch.
    """
    print("=" * 60)
    print("K-Nearest Neighbors from Scratch - Demo")
    print("=" * 60)
    
    # Load dataset
    print("\n[1] Loading dataset...")
    data = pd.read_csv("../../Datasets/knn_data.csv")
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {data.columns.tolist()[:-1]}")
    print(f"Target: {data.columns.tolist()[-1]}")
    print(f"Class distribution: {data.iloc[:, -1].value_counts().to_dict()}")
    
    # Prepare data
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Split data (80-20 train-test split)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Feature scaling (standardization) - crucial for distance-based algorithms
    print("\n[2] Scaling features...")
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std
    
    # Find optimal K using cross-validation
    print("\n[3] Finding optimal K...")
    k_values = [1, 3, 5, 7, 9, 11, 15]
    accuracies = []
    
    # Simple validation split (last 20% of training data)
    val_split = int(0.8 * len(X_train_scaled))
    X_val = X_train_scaled[val_split:]
    y_val = y_train[val_split:]
    X_train_cv = X_train_scaled[:val_split]
    y_train_cv = y_train[:val_split]
    
    for k in k_values:
        model = KNNScratch(k=k, distance_metric='euclidean', weights='uniform')
        model.fit(X_train_cv, y_train_cv)
        accuracy = model.score(X_val, y_val)
        accuracies.append(accuracy)
        print(f"K={k:2d}, Validation Accuracy: {accuracy:.4f}")
    
    optimal_k = k_values[np.argmax(accuracies)]
    print(f"\nOptimal K: {optimal_k} with accuracy: {max(accuracies):.4f}")
    
    # Train final model with optimal K
    print(f"\n[4] Training final model with K={optimal_k}...")
    model = KNNScratch(k=optimal_k, distance_metric='euclidean', weights='distance')
    model.fit(X_train_scaled, y_train)
    print("Model training complete!")
    
    # Make predictions
    print("\n[5] Making predictions...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Evaluate model
    print("\n[6] Model Evaluation")
    print("-" * 60)
    train_acc = np.mean(y_train_pred == y_train)
    test_acc = np.mean(y_test_pred == y_test)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    
    # Calculate precision, recall, F1 for each class
    classes = np.unique(y_test)
    print(f"\nPer-class metrics:")
    for cls in classes:
        # True Positives, False Positives, False Negatives
        tp = np.sum((y_test_pred == cls) & (y_test == cls))
        fp = np.sum((y_test_pred == cls) & (y_test != cls))
        fn = np.sum((y_test_pred != cls) & (y_test == cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  Class {cls}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    # Confusion matrix
    print("\n[7] Confusion Matrix (Test Set)")
    print("-" * 60)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for true_label, pred_label in zip(y_test, y_test_pred):
        true_idx = np.where(classes == true_label)[0][0]
        pred_idx = np.where(classes == pred_label)[0][0]
        cm[true_idx, pred_idx] += 1
    
    print("       ", end="")
    for c in classes:
        print(f"Pred {c:<4}", end=" ")
    print()
    for i, c in enumerate(classes):
        print(f"Act {c:<3}", end=" ")
        for j in range(len(classes)):
            print(f"{cm[i, j]:<9}", end=" ")
        print()
    
    # Sample predictions with neighbor analysis
    print("\n[8] Sample Predictions (First 5 test samples)")
    print("-" * 60)
    print(f"{'Actual':<10} {'Predicted':<12} {'Correct':<10}")
    print("-" * 60)
    for i in range(min(5, len(y_test))):
        correct = "Yes" if y_test[i] == y_test_pred[i] else "No"
        print(f"{y_test[i]:<10} {y_test_pred[i]:<12} {correct:<10}")
    
    # Compare distance metrics
    print("\n[9] Comparing Distance Metrics")
    print("-" * 60)
    for metric in ['euclidean', 'manhattan']:
        model_compare = KNNScratch(k=optimal_k, distance_metric=metric, weights='uniform')
        model_compare.fit(X_train_scaled, y_train)
        acc = model_compare.score(X_test_scaled, y_test)
        print(f"{metric.capitalize():12s}: {acc:.4f}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
