"""
Linear Regression Implementation using Scikit-learn
Production-ready implementation with best practices
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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


def train_model(X_train, y_train):
    """
    Train Linear Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        Trained model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance on train and test sets.
    
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
    
    # Calculate metrics
    metrics = {
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred)
    }
    
    return metrics, y_test_pred


def display_results(model, metrics, y_test, y_test_pred, feature_names=None):
    """
    Display comprehensive results.
    
    Args:
        model: Trained model
        metrics: Dictionary of evaluation metrics
        y_test: True test values
        y_test_pred: Predicted test values
        feature_names: List of feature names (optional)
    """
    print("=" * 70)
    print("Linear Regression with Scikit-learn - Results")
    print("=" * 70)
    
    # Model coefficients
    print("\n[1] Model Parameters")
    print("-" * 70)
    print(f"Intercept (bias): {model.intercept_:.4f}")
    print("\nCoefficients (weights):")
    if feature_names:
        for name, coef in zip(feature_names, model.coef_):
            print(f"  {name}: {coef:.4f}")
    else:
        for i, coef in enumerate(model.coef_):
            print(f"  Feature {i + 1}: {coef:.4f}")
    
    # Performance metrics
    print("\n[2] Performance Metrics")
    print("-" * 70)
    print("Training Set:")
    print(f"  R² Score: {metrics['train_r2']:.4f}")
    print(f"  MSE: {metrics['train_mse']:.4f}")
    print(f"  RMSE: {metrics['train_rmse']:.4f}")
    print(f"  MAE: {metrics['train_mae']:.4f}")
    
    print("\nTest Set:")
    print(f"  R² Score: {metrics['test_r2']:.4f}")
    print(f"  MSE: {metrics['test_mse']:.4f}")
    print(f"  RMSE: {metrics['test_rmse']:.4f}")
    print(f"  MAE: {metrics['test_mae']:.4f}")
    
    # Overfitting check
    print("\n[3] Model Analysis")
    print("-" * 70)
    r2_diff = metrics['train_r2'] - metrics['test_r2']
    if r2_diff < 0.05:
        status = "Good fit - minimal overfitting"
    elif r2_diff < 0.1:
        status = "Acceptable fit - slight overfitting"
    else:
        status = "Potential overfitting detected"
    print(f"R² difference: {r2_diff:.4f} - {status}")
    
    # Sample predictions
    print("\n[4] Sample Predictions (First 5 test samples)")
    print("-" * 70)
    print(f"{'Actual':<15} {'Predicted':<15} {'Error':<15} {'% Error':<15}")
    print("-" * 70)
    for i in range(min(5, len(y_test))):
        actual = y_test[i]
        predicted = y_test_pred[i]
        error = abs(actual - predicted)
        pct_error = (error / actual * 100) if actual != 0 else 0
        print(f"{actual:<15.2f} {predicted:<15.2f} {error:<15.2f} {pct_error:<15.2f}")
    
    print("\n" + "=" * 70)


def make_prediction(model, scaler, new_data):
    """
    Make predictions on new data.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        new_data: New data points (2D array or list)
    
    Returns:
        Predictions
    """
    new_data = np.array(new_data).reshape(-1, model.n_features_in_)
    new_data_scaled = scaler.transform(new_data)
    predictions = model.predict(new_data_scaled)
    return predictions


def main():
    """
    Main execution function.
    """
    # File path
    filepath = "../../Datasets/sample_data.csv"
    
    print("\n[1] Loading and preparing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data(filepath)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    
    print("\n[2] Training model...")
    model = train_model(X_train, y_train)
    print("Model training complete!")
    
    print("\n[3] Evaluating model...")
    metrics, y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    print("\n[4] Displaying results...")
    display_results(model, metrics, y_test, y_test_pred)
    
    # Example: Making predictions on new data
    print("\n[5] Example: Predicting on new data")
    print("-" * 70)
    # Create sample new data (adjust dimensions based on your features)
    new_data = X_test[:3]  # Using first 3 test samples as example
    predictions = make_prediction(model, scaler, new_data)
    print(f"New data predictions: {predictions}")
    
    print("\nProcess complete!")


if __name__ == "__main__":
    main()
