"""
Support Vector Machine Implementation from Scratch
Using only NumPy - implements SVM with gradient descent (simplified version)
Note: This is a simplified linear SVM. Full SVM with kernels requires
quadratic programming solvers which are complex to implement from scratch.
"""

import numpy as np
import pandas as pd


class SVMScratch:
    """
    Linear Support Vector Machine implemented from scratch using gradient descent.
    This is a simplified implementation for educational purposes.
    
    Attributes:
        learning_rate: Learning rate for gradient descent
        lambda_param: Regularization parameter (inverse of C)
        n_iterations: Number of training iterations
        weights: Model weights
        bias: Model bias
    """
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        """
        Initialize SVM classifier.
        
        Args:
            learning_rate: Learning rate (default: 0.001)
            lambda_param: Regularization strength (default: 0.01)
            n_iterations: Number of iterations (default: 1000)
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """
        Train SVM using gradient descent with hinge loss.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels (must be -1 or 1), shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        
        # Convert labels to -1 and 1 if they are 0 and 1
        y_ = np.where(y <= 0, -1, 1)
        
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training using gradient descent
        for iteration in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # Condition: y_i * (w · x_i + b) >= 1
                condition = y_[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                
                if condition:
                    # If correctly classified, only update regularization term
                    # gradient: λw
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    # If misclassified or within margin
                    # gradient: λw - y_i * x_i
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx])
                    )
                    self.bias -= self.learning_rate * y_[idx]
            
            # Print progress
            if (iteration + 1) % 100 == 0:
                loss = self._calculate_hinge_loss(X, y_)
                print(f"Iteration {iteration + 1}/{self.n_iterations}, Loss: {loss:.4f}")
    
    def _calculate_hinge_loss(self, X, y):
        """
        Calculate hinge loss: max(0, 1 - y * (w·x + b))
        
        Args:
            X: Features
            y: Labels (-1 or 1)
        
        Returns:
            Average hinge loss
        """
        if self.weights is None or self.bias is None:
            return 0.0
        distances = 1 - y * (np.dot(X, self.weights) + self.bias)
        hinge_loss = np.maximum(0, distances)
        # Total loss = regularization + hinge loss
        loss = self.lambda_param * np.dot(self.weights, self.weights) + np.mean(hinge_loss)
        return loss
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Features, shape (n_samples, n_features)
        
        Returns:
            Predicted labels (0 or 1)
        """
        X = np.array(X)
        if self.weights is None or self.bias is None:
            return np.zeros(len(X), dtype=int)
        linear_output = np.dot(X, self.weights) + self.bias
        # Return 1 if >= 0, else 0
        return np.where(linear_output >= 0, 1, 0)
    
    def score(self, X, y):
        """
        Calculate accuracy.
        
        Args:
            X: Features
            y: True labels
        
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def main():
    """
    Demonstration of Linear SVM from scratch.
    """
    print("=" * 60)
    print("Linear SVM from Scratch - Demo")
    print("=" * 60)
    
    # Load dataset
    print("\n[1] Loading dataset...")
    data = pd.read_csv("../../Datasets/svm_data.csv")
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
    
    # Feature scaling - crucial for SVM
    print("\n[2] Scaling features...")
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std
    
    # Train model
    print("\n[3] Training SVM model...")
    model = SVMScratch(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
    model.fit(X_train_scaled, y_train)
    print("Model training complete!")
    
    # Make predictions
    print("\n[4] Making predictions...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Evaluate model
    print("\n[5] Model Evaluation")
    print("-" * 60)
    train_acc = np.mean(y_train_pred == y_train)
    test_acc = np.mean(y_test_pred == y_test)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    
    # Calculate metrics
    tp = np.sum((y_test_pred == 1) & (y_test == 1))
    fp = np.sum((y_test_pred == 1) & (y_test == 0))
    fn = np.sum((y_test_pred == 0) & (y_test == 1))
    tn = np.sum((y_test_pred == 0) & (y_test == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion matrix
    print("\n[6] Confusion Matrix (Test Set)")
    print("-" * 60)
    print(f"                Predicted 0    Predicted 1")
    print(f"Actual 0        {tn:<14} {fp:<14}")
    print(f"Actual 1        {fn:<14} {tp:<14}")
    
    # Model parameters
    print("\n[7] Learned Parameters")
    print("-" * 60)
    print(f"Bias: {model.bias:.4f}")
    print(f"Weights: {model.weights}")
    if model.weights is not None:
        print(f"Weight magnitude: {np.linalg.norm(model.weights):.4f}")
    
    # Sample predictions
    print("\n[8] Sample Predictions (First 10 test samples)")
    print("-" * 60)
    print(f"{'Actual':<10} {'Predicted':<12} {'Decision Value':<18} {'Correct':<10}")
    print("-" * 60)
    for i in range(min(10, len(y_test))):
        if model.weights is None or model.bias is None:
            decision_val = 0.0
        else:
            decision_val = np.dot(X_test_scaled[i], model.weights) + model.bias
        correct = "Yes" if y_test[i] == y_test_pred[i] else "No"
        print(f"{y_test[i]:<10} {y_test_pred[i]:<12} {decision_val:<18.4f} {correct:<10}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nNote: This is a simplified linear SVM implementation.")
    print("For production use, consider scikit-learn's SVM with kernel support.")


if __name__ == "__main__":
    main()
