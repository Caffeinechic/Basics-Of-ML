"""
Decision Tree Classifier Implementation from Scratch
Using only NumPy for educational purposes
"""

import numpy as np
import pandas as pd
from collections import Counter


class Node: 
    """
    Node class for Decision Tree.
    
    Attributes:
        feature: Feature index to split on
        threshold: Threshold value for splitting
        left: Left child node
        right: Right child node
        value: Predicted class (for leaf nodes)
    """
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        """Check if node is a leaf."""
        return self.value is not None


class DecisionTreeScratch:
    """
    Decision Tree Classifier implemented from scratch.
    
    Attributes:
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split a node
        n_features: Number of features to consider for best split
        root: Root node of the tree
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        """
        Initialize Decision Tree.
        
        Args:
            max_depth: Maximum depth of tree (default: 10)
            min_samples_split: Minimum samples to split (default: 2)
            n_features: Number of features to consider (default: all)
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
    
    def fit(self, X, y):
        """
        Build decision tree from training data.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Target labels, shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        
        # Set n_features if not provided
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        
        # Build tree recursively
        self.root = self._build_tree(X, y)
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        
        Args:
            X: Features for current node
            y: Labels for current node
            depth: Current depth in tree
        
        Returns:
            Node object
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_classes == 1 or 
            n_samples < self.min_samples_split):
            # Create leaf node with most common class
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find best split
        feature_indices = np.random.choice(n_features, self.n_features, replace=False)
        best_feature, best_threshold = self._best_split(X, y, feature_indices)
        
        # If no valid split found, create leaf
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Split data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, 
                   left=left_child, right=right_child)
    
    def _best_split(self, X, y, feature_indices):
        """
        Find the best feature and threshold to split on.
        
        Args:
            X: Features
            y: Labels
            feature_indices: Indices of features to consider
        
        Returns:
            best_feature, best_threshold
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                # Calculate information gain
                gain = self._information_gain(X, y, feature_idx, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _information_gain(self, X, y, feature_idx, threshold):
        """
        Calculate information gain for a split.
        
        Args:
            X: Features
            y: Labels
            feature_idx: Feature to split on
            threshold: Threshold value
        
        Returns:
            Information gain value
        """
        # Parent entropy
        parent_entropy = self._entropy(y)
        
        # Split data
        left_indices = X[:, feature_idx] <= threshold
        right_indices = X[:, feature_idx] > threshold
        
        # If split doesn't divide data, return 0 gain
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return 0
        
        # Calculate weighted average of child entropies
        n = len(y)
        n_left = np.sum(left_indices)
        n_right = np.sum(right_indices)
        
        entropy_left = self._entropy(y[left_indices])
        entropy_right = self._entropy(y[right_indices])
        
        child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right
        
        # Information gain
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    def _entropy(self, y):
        """
        Calculate entropy of labels.
        
        Args:
            y: Labels
        
        Returns:
            Entropy value
        """
        proportions = np.bincount(y) / len(y)
        # Avoid log(0)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy
    
    def _most_common_label(self, y):
        """
        Find the most common label.
        
        Args:
            y: Labels
        
        Returns:
            Most common label
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """
        Predict class labels for samples.
        
        Args:
            X: Features, shape (n_samples, n_features)
        
        Returns:
            Predicted labels, shape (n_samples,)
        """
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """
        Traverse tree to make prediction for single sample.
        
        Args:
            x: Single sample features
            node: Current node
        
        Returns:
            Predicted label
        """
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def accuracy_score(self, y_true, y_pred):
        """Calculate accuracy."""
        return np.mean(y_true == y_pred)


def main():
    """
    Demonstration of Decision Tree from scratch.
    """
    print("=" * 60)
    print("Decision Tree Classifier from Scratch - Demo")
    print("=" * 60)
    
    # Load dataset
    print("\n[1] Loading dataset...")
    data = pd.read_csv("../../Datasets/tree_data.csv")
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
    
    # Train model
    print("\n[2] Training model...")
    model = DecisionTreeScratch(max_depth=5, min_samples_split=2)
    model.fit(X_train, y_train)
    print("Model training complete!")
    
    # Make predictions
    print("\n[3] Making predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate model
    print("\n[4] Model Evaluation")
    print("-" * 60)
    train_acc = model.accuracy_score(y_train, y_train_pred)
    test_acc = model.accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    
    # Confusion matrix
    print("\n[5] Confusion Matrix (Test Set)")
    print("-" * 60)
    classes = np.unique(np.array(y))
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for true_label, pred_label in zip(y_test, y_test_pred):
        cm[true_label, pred_label] += 1
    
    print("      ", end="")
    for c in classes:
        print(f"Pred {c:<4}", end=" ")
    print()
    for i, c in enumerate(classes):
        print(f"Act {c:<2}", end=" ")
        for j in range(len(classes)):
            print(f"{cm[i, j]:<9}", end=" ")
        print()
    
    # Sample predictions
    print("\n[6] Sample Predictions (First 10 test samples)")
    print("-" * 60)
    print(f"{'Actual':<10} {'Predicted':<10} {'Correct':<10}")
    print("-" * 60)
    for i in range(min(10, len(y_test))):
        correct = "Yes" if y_test[i] == y_test_pred[i] else "No"
        print(f"{y_test[i]:<10} {y_test_pred[i]:<10} {correct:<10}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
