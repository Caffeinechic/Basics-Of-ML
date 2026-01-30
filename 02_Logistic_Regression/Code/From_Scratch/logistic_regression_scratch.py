"""
Logistic Regression Implementation from Scratch
Using only NumPy for educational purposes
"""

import numpy as np
import pandas as pd


class LogisticRegressionScratch:
    """
    Logistic Regression implemented from scratch using gradient descent.
    
    Attributes:
        learning_rate (float): Step size for gradient descent
        n_iterations (int): Number of training iterations
        weights (np.ndarray): Model coefficients
        bias (float): Model intercept
        cost_history (list): Training loss at each iteration
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize the Logistic Regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent (default: 0.01)
            n_iterations: Number of iterations for training (default: 1000)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def sigmoid(self, z):
        """
        Sigmoid activation function.
        
        Args:
            z: Linear combination of features and weights
        
        Returns:
            Sigmoid output (probability between 0 and 1)
        """
        # Clip z to prevent overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Train the model using gradient descent.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Target values (binary: 0 or 1), shape (n_samples,)
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Get dimensions
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass: compute linear model and apply sigmoid
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            # Compute cost (Binary Cross-Entropy / Log Loss)
            epsilon = 1e-15  # To avoid log(0)
            y_predicted_clipped = np.clip(y_predicted, epsilon, 1 - epsilon)
            cost = -(1 / n_samples) * np.sum(
                y * np.log(y_predicted_clipped) + 
                (1 - y) * np.log(1 - y_predicted_clipped)
            )
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{self.n_iterations}, Cost: {cost:.4f}")
    
    def predict_proba(self, X):
        """
        Predict probabilities for samples.
        
        Args:
            X: Features, shape (n_samples, n_features)
        
        Returns:
            Predicted probabilities, shape (n_samples,)
        """
        X = np.array(X)
        if self.weights is None or self.bias is None:
            return np.zeros(len(X))
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels for samples.
        
        Args:
            X: Features, shape (n_samples, n_features)
            threshold: Decision threshold (default: 0.5)
        
        Returns:
            Predicted class labels (0 or 1), shape (n_samples,)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def accuracy_score(self, y_true, y_pred):
        """
        Calculate accuracy score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Accuracy value
        """
        return np.mean(y_true == y_pred)
    
    def precision_score(self, y_true, y_pred):
        """
        Calculate precision score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Precision value
        """
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / predicted_positives if predicted_positives > 0 else 0
    
    def recall_score(self, y_true, y_pred):
        """
        Calculate recall score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Recall value
        """
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / actual_positives if actual_positives > 0 else 0
    
    def f1_score(self, y_true, y_pred):
        """
        Calculate F1 score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            F1 score value
        """
        precision = self.precision_score(y_true, y_pred)
        recall = self.recall_score(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    def confusion_matrix(self, y_true, y_pred):
        """
        Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            2x2 confusion matrix [[TN, FP], [FN, TP]]
        """
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tp = np.sum((y_pred == 1) & (y_true == 1))
        return np.array([[tn, fp], [fn, tp]])


def main():
    """
    Demonstration of Logistic Regression from scratch.
    """
    print("=" * 60)
    print("Logistic Regression from Scratch - Demo")
    print("=" * 60)
    
    # Load dataset
    print("\n[1] Loading dataset...")
    data = pd.read_csv("../../Datasets/classification_data.csv")
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
    
    # Feature scaling (standardization)
    print("\n[2] Scaling features...")
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std
    
    # Train model
    print("\n[3] Training model...")
    model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    print("\n[4] Making predictions...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)
    
    # Evaluate model
    print("\n[5] Model Evaluation")
    print("-" * 60)
    
    # Training metrics
    train_acc = model.accuracy_score(y_train, y_train_pred)
    train_precision = model.precision_score(y_train, y_train_pred)
    train_recall = model.recall_score(y_train, y_train_pred)
    train_f1 = model.f1_score(y_train, y_train_pred)
    
    # Testing metrics
    test_acc = model.accuracy_score(y_test, y_test_pred)
    test_precision = model.precision_score(y_test, y_test_pred)
    test_recall = model.recall_score(y_test, y_test_pred)
    test_f1 = model.f1_score(y_test, y_test_pred)
    
    print("Training Set:")
    print(f"  Accuracy: {train_acc:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall: {train_recall:.4f}")
    print(f"  F1 Score: {train_f1:.4f}")
    
    print("\nTest Set:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    
    # Confusion matrix
    print("\n[6] Confusion Matrix (Test Set)")
    print("-" * 60)
    cm = model.confusion_matrix(y_test, y_test_pred)
    print(f"                Predicted 0    Predicted 1")
    print(f"Actual 0        {cm[0, 0]:<14} {cm[0, 1]:<14}")
    print(f"Actual 1        {cm[1, 0]:<14} {cm[1, 1]:<14}")
    
    # Display learned parameters
    print("\n[7] Learned Parameters")
    print("-" * 60)
    print(f"Bias (intercept): {model.bias:.4f}")
    print("Weights (coefficients):")
    if model.weights is not None:
        for i, weight in enumerate(model.weights):
            print(f"  Feature {i + 1}: {weight:.4f}")
    
    # Sample predictions
    print("\n[8] Sample Predictions (First 5 test samples)")
    print("-" * 60)
    print(f"{'Actual':<10} {'Predicted':<10} {'Probability':<15}")
    print("-" * 60)
    for i in range(min(5, len(y_test))):
        print(f"{y_test[i]:<10} {y_test_pred[i]:<10} {y_test_proba[i]:<15.4f}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
