"""
Gaussian Naive Bayes Implementation from Scratch
Using only NumPy - implements Bayes theorem and probability calculations
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Dict, Any


class GaussianNaiveBayesScratch:
    """
    Gaussian Naive Bayes implemented from scratch.
    
    Attributes:
        classes: Unique class labels
        class_priors: Prior probabilities for each class
        means: Mean of each feature for each class
        vars: Variance of each feature for each class
    """
    
    def __init__(self):
        """Initialize Gaussian Naive Bayes classifier."""
        self.classes: NDArray[np.generic] = np.array([])
        self.class_priors: Dict[Any, float] = {}
        self.means: Dict[Any, NDArray[np.floating]] = {}
        self.vars: Dict[Any, NDArray[np.floating]] = {}
    
    def fit(self, X, y):
        """
        Train the classifier by calculating statistics.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        n_samples = len(y)
        
        # Calculate prior probabilities and statistics for each class
        for cls in self.classes:
            # Get samples belonging to this class
            X_cls = X[y == cls]
            
            # Prior probability: P(Class)
            self.class_priors[cls] = len(X_cls) / n_samples
            
            # Calculate mean and variance for each feature
            # Mean: μ = (1/n) Σ xᵢ
            self.means[cls] = np.mean(X_cls, axis=0)
            
            # Variance: σ² = (1/n) Σ (xᵢ - μ)²
            self.vars[cls] = np.var(X_cls, axis=0) + 1e-9  # Add small value to avoid division by zero
    
    def _calculate_likelihood(self, x, mean, var):
        """
        Calculate Gaussian probability density function.
        P(x | class) = (1 / √(2πσ²)) × exp(-(x-μ)² / (2σ²))
        
        Args:
            x: Feature value
            mean: Mean of feature for a class
            var: Variance of feature for a class
        
        Returns:
            Likelihood probability
        """
        # Gaussian PDF formula
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def _calculate_posterior(self, x):
        """
        Calculate posterior probability for each class.
        P(Class | X) ∝ P(Class) × ∏ P(xᵢ | Class)
        
        Args:
            x: Single sample features
        
        Returns:
            Dictionary of log posterior probabilities for each class
        """
        posteriors = {}
        
        for cls in self.classes:
            # Start with log prior: log P(Class)
            log_prior = np.log(self.class_priors[cls])
            
            # Calculate log likelihood for each feature: Σ log P(xᵢ | Class)
            # Using log to avoid underflow from multiplying small probabilities
            likelihoods = self._calculate_likelihood(x, self.means[cls], self.vars[cls])
            log_likelihood = np.sum(np.log(likelihoods + 1e-10))  # Add small value to avoid log(0)
            
            # Posterior: log P(Class) + Σ log P(xᵢ | Class)
            posteriors[cls] = log_prior + log_likelihood
        
        return posteriors
    
    def predict_single(self, x):
        """
        Predict class for a single sample.
        
        Args:
            x: Single sample features
        
        Returns:
            Predicted class label
        """
        posteriors = self._calculate_posterior(x)
        # Return class with maximum posterior probability
        return max(posteriors, key=lambda k: posteriors[k])
    
    def predict(self, X):
        """
        Predict classes for multiple samples.
        
        Args:
            X: Features, shape (n_samples, n_features)
        
        Returns:
            Predicted labels, shape (n_samples,)
        """
        X = np.array(X)
        predictions = [self.predict_single(x) for x in X]
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples.
        
        Args:
            X: Features, shape (n_samples, n_features)
        
        Returns:
            Probability matrix, shape (n_samples, n_classes)
        """
        X = np.array(X)
        probabilities = []
        
        for x in X:
            posteriors = self._calculate_posterior(x)
            
            # Convert log probabilities to probabilities using softmax
            log_probs = np.array([posteriors[cls] for cls in self.classes])
            # Softmax: exp(x) / sum(exp(x))
            # Subtract max for numerical stability
            log_probs = log_probs - np.max(log_probs)
            probs = np.exp(log_probs) / np.sum(np.exp(log_probs))
            probabilities.append(probs)
        
        return np.array(probabilities)
    
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
    Demonstration of Gaussian Naive Bayes from scratch.
    """
    print("=" * 60)
    print("Gaussian Naive Bayes from Scratch - Demo")
    print("=" * 60)
    
    # Load dataset
    print("\n[1] Loading dataset...")
    data = pd.read_csv("../../Datasets/naive_bayes_data.csv")
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
    model = GaussianNaiveBayesScratch()
    model.fit(X_train, y_train)
    print("Model training complete!")
    
    # Display learned parameters
    print("\n[3] Learned Parameters")
    print("-" * 60)
    for cls in model.classes:
        print(f"\nClass {cls}:")
        print(f"  Prior probability: {model.class_priors[cls]:.4f}")
        print(f"  Feature means: {model.means[cls]}")
        print(f"  Feature variances: {model.vars[cls]}")
    
    # Make predictions
    print("\n[4] Making predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    
    # Evaluate model
    print("\n[5] Model Evaluation")
    print("-" * 60)
    train_acc = np.mean(y_train_pred == y_train)
    test_acc = np.mean(y_test_pred == y_test)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    
    # Calculate per-class metrics
    classes = model.classes
    print(f"\nPer-class metrics:")
    for cls in classes:
        tp = np.sum((y_test_pred == cls) & (y_test == cls))
        fp = np.sum((y_test_pred == cls) & (y_test != cls))
        fn = np.sum((y_test_pred != cls) & (y_test == cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  Class {cls}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    # Confusion matrix
    print("\n[6] Confusion Matrix (Test Set)")
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
    
    # Sample predictions
    print("\n[7] Sample Predictions (First 5 test samples)")
    print("-" * 60)
    print(f"{'Actual':<10} {'Predicted':<12} {'Probabilities':<30} {'Correct':<10}")
    print("-" * 60)
    for i in range(min(5, len(y_test))):
        probs_str = ", ".join([f"{p:.3f}" for p in y_test_proba[i]])
        correct = "Yes" if y_test[i] == y_test_pred[i] else "No"
        print(f"{y_test[i]:<10} {y_test_pred[i]:<12} [{probs_str}]  {correct:<10}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
