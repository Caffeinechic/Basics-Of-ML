"""
Linear Regression Implementation from Scratch
Using only NumPy for educational purposes
"""

import numpy as np
import pandas as pd


class LinearRegressionScratch:
    """
    Linear Regression implemented from scratch using gradient descent.
    
    Attributes:
        learning_rate (float): Step size for gradient descent
        n_iterations (int): Number of training iterations
        weights (np.ndarray): Model coefficients
        bias (float): Model intercept
        cost_history (list): Training loss at each iteration
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize the Linear Regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent (default: 0.01)
            n_iterations: Number of iterations for training (default: 1000)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """
        Train the model using gradient descent.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Target values, shape (n_samples,)
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
            # Forward pass: compute predictions
            y_predicted = self.predict(X)
            
            # Compute cost (Mean Squared Error)
            cost = (1 / (2 * n_samples)) * np.sum((y_predicted - y) ** 2)
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
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features, shape (n_samples, n_features)
        
        Returns:
            Predictions, shape (n_samples,)
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model must be trained before making predictions. Call fit() first.")
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X, y):
        """
        Calculate R-squared score.
        
        Args:
            X: Features, shape (n_samples, n_features)
            y: True values, shape (n_samples,)
        
        Returns:
            R-squared score
        """
        y = np.array(y)
        y_pred = self.predict(X)
        
        # Total sum of squares
        ss_total = np.sum((y - np.mean(y)) ** 2)
        
        # Residual sum of squares
        ss_residual = np.sum((y - y_pred) ** 2)
        
        # R-squared
        r2 = 1 - (ss_residual / ss_total)
        return r2
    
    def mean_squared_error(self, y_true, y_pred):
        """
        Calculate Mean Squared Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            MSE value
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def mean_absolute_error(self, y_true, y_pred):
        """
        Calculate Mean Absolute Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            MAE value
        """
        return np.mean(np.abs(y_true - y_pred))


def main():
    """
    Demonstration of Linear Regression from scratch.
    """
    print("=" * 60)
    print("Linear Regression from Scratch - Demo")
    print("=" * 60)
    
    # Load dataset
    print("\n[1] Loading dataset...")
    data = pd.read_csv("../../Datasets/sample_data.csv")
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {data.columns.tolist()[:-1]}")
    print(f"Target: {data.columns.tolist()[-1]}")
    
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
    model = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    print("\n[4] Making predictions...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Evaluate model
    print("\n[5] Model Evaluation")
    print("-" * 60)
    train_r2 = model.score(X_train_scaled, y_train)
    test_r2 = model.score(X_test_scaled, y_test)
    
    train_mse = model.mean_squared_error(y_train, y_train_pred)
    test_mse = model.mean_squared_error(y_test, y_test_pred)
    
    train_mae = model.mean_absolute_error(y_train, y_train_pred)
    test_mae = model.mean_absolute_error(y_test, y_test_pred)
    
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Testing R² Score: {test_r2:.4f}")
    print(f"\nTraining MSE: {train_mse:.4f}")
    print(f"Testing MSE: {test_mse:.4f}")
    print(f"\nTraining MAE: {train_mae:.4f}")
    print(f"Testing MAE: {test_mae:.4f}")
    
    # Display learned parameters
    print("\n[6] Learned Parameters")
    print("-" * 60)
    print(f"Bias (intercept): {model.bias:.4f}")
    print("Weights (coefficients):")
    if model.weights is not None:
        for i, weight in enumerate(model.weights):
            print(f"  Feature {i + 1}: {weight:.4f}")
    
    # Sample predictions
    print("\n[7] Sample Predictions (First 5 test samples)")
    print("-" * 60)
    print(f"{'Actual':<12} {'Predicted':<12} {'Error':<12}")
    print("-" * 60)
    for i in range(min(5, len(y_test))):
        error = abs(y_test[i] - y_test_pred[i])
        print(f"{y_test[i]:<12.2f} {y_test_pred[i]:<12.2f} {error:<12.2f}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
