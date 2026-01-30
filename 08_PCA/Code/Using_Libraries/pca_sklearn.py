"""
Principal Component Analysis Implementation using Scikit-learn
Production-ready implementation with advanced features
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(filepath):
    """
    Load and prepare data for PCA.
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        X_scaled: Standardized features
        scaler: Fitted StandardScaler object
    """
    # Load data
    data = pd.read_csv(filepath)
    X = data.values
    
    # Standardization - critical for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler


def determine_optimal_components(X, variance_threshold=0.95):
    """
    Determine optimal number of components.
    
    Args:
        X: Scaled features
        variance_threshold: Desired cumulative variance (default: 0.95)
    
    Returns:
        Optimal number of components
    """
    # Fit PCA with all components
    pca_full = PCA()
    pca_full.fit(X)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    print("\n[Explained Variance Analysis]")
    print("-" * 70)
    print(f"{'Component':<12} {'Variance':<15} {'Cumulative':<15}")
    print("-" * 70)
    
    for i, (var, cum_var) in enumerate(zip(pca_full.explained_variance_ratio_, cumulative_variance)):
        print(f"PC{i + 1:<10} {var:<15.4f} {cum_var:<15.4f}")
        if i >= 9:  # Show first 10 components
            break
    
    # Find optimal number
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    print(f"\nOptimal components for {variance_threshold:.0%} variance: {n_components}")
    print(f"Actual variance retained: {cumulative_variance[n_components - 1]:.4f}")
    
    return n_components, pca_full


def apply_pca(X, n_components):
    """
    Apply PCA transformation.
    
    Args:
        X: Scaled features
        n_components: Number of components to keep
    
    Returns:
        Fitted PCA model and transformed data
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    
    return pca, X_pca


def analyze_components(pca, feature_names=None):
    """
    Analyze principal components.
    
    Args:
        pca: Fitted PCA model
        feature_names: List of original feature names (optional)
    """
    print("\n[Principal Component Analysis]")
    print("-" * 70)
    
    # Component loadings
    print("\nComponent Loadings (first 3 components):")
    components_to_show = min(3, pca.n_components_)
    
    if feature_names:
        print(f"{'Feature':<15}", end="")
        for i in range(components_to_show):
            print(f"PC{i + 1:<13}", end="")
        print()
        print("-" * 70)
        
        for i, feature in enumerate(feature_names):
            print(f"{feature:<15}", end="")
            for j in range(components_to_show):
                print(f"{pca.components_[j, i]:<15.4f}", end="")
            print()
    else:
        for i in range(components_to_show):
            print(f"PC{i + 1}: {pca.components_[i]}")
    
    # Top contributing features for each component
    print("\nTop Contributing Features (by absolute loading):")
    for i in range(min(3, pca.n_components_)):
        top_indices = np.argsort(np.abs(pca.components_[i]))[-3:][::-1]
        print(f"\nPC{i + 1}:")
        for idx in top_indices:
            feature_name = feature_names[idx] if feature_names else f"Feature {idx}"
            print(f"  {feature_name}: {pca.components_[i, idx]:.4f}")


def evaluate_pca(pca, X_original, X_pca):
    """
    Evaluate PCA performance.
    
    Args:
        pca: Fitted PCA model
        X_original: Original data
        X_pca: Transformed data
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Reconstruction
    X_reconstructed = pca.inverse_transform(X_pca)
    
    # Reconstruction error
    mse = np.mean((X_original - X_reconstructed) ** 2)
    mae = np.mean(np.abs(X_original - X_reconstructed))
    
    metrics = {
        'reconstruction_mse': mse,
        'reconstruction_mae': mae,
        'explained_variance': np.sum(pca.explained_variance_ratio_),
        'n_components': pca.n_components_,
        'compression_ratio': pca.n_components_ / X_original.shape[1]
    }
    
    return metrics, X_reconstructed


def display_results(pca, metrics, X_original, X_pca):
    """
    Display comprehensive results.
    
    Args:
        pca: Fitted PCA model
        metrics: Evaluation metrics
        X_original: Original data
        X_pca: Transformed data
    """
    print("=" * 70)
    print("Principal Component Analysis with Scikit-learn - Results")
    print("=" * 70)
    
    # Model information
    print("\n[1] Model Information")
    print("-" * 70)
    print(f"Original dimensions: {X_original.shape[1]}")
    print(f"Reduced dimensions: {pca.n_components_}")
    print(f"Compression ratio: {metrics['compression_ratio']:.2%}")
    print(f"Number of samples: {X_original.shape[0]}")
    
    # Variance explained
    print("\n[2] Variance Explained")
    print("-" * 70)
    for i, var in enumerate(pca.explained_variance_ratio_):
        cum_var = np.sum(pca.explained_variance_ratio_[:i + 1])
        print(f"PC{i + 1}: {var:.4f} (Cumulative: {cum_var:.4f})")
    
    print(f"\nTotal variance explained: {metrics['explained_variance']:.4f}")
    print(f"Variance lost: {1 - metrics['explained_variance']:.4f}")
    
    # Reconstruction quality
    print("\n[3] Reconstruction Quality")
    print("-" * 70)
    print(f"Mean Squared Error: {metrics['reconstruction_mse']:.6f}")
    print(f"Mean Absolute Error: {metrics['reconstruction_mae']:.6f}")
    
    relative_error = metrics['reconstruction_mse'] / np.var(X_original)
    print(f"Relative MSE: {relative_error:.6f}")
    
    if relative_error < 0.05:
        quality = "Excellent"
    elif relative_error < 0.1:
        quality = "Good"
    elif relative_error < 0.2:
        quality = "Fair"
    else:
        quality = "Poor"
    print(f"Reconstruction quality: {quality}")
    
    # Component importance visualization
    print("\n[4] Component Importance (Visual)")
    print("-" * 70)
    for i in range(min(10, len(pca.explained_variance_ratio_))):
        bar_length = int(pca.explained_variance_ratio_[i] * 50)
        bar = '#' * bar_length
        print(f"PC{i + 1:>2}: {bar} {pca.explained_variance_ratio_[i]:.4f}")
    
    # Singular values
    print("\n[5] Singular Values")
    print("-" * 70)
    print(f"Singular values: {pca.singular_values_[:min(5, len(pca.singular_values_))]}")
    
    # Sample transformation
    print("\n[6] Sample Transformation (First 3 samples)")
    print("-" * 70)
    for i in range(min(3, len(X_pca))):
        print(f"\nSample {i + 1}:")
        print(f"  Original dimensions: {X_original.shape[1]}")
        print(f"  PCA dimensions: {X_pca.shape[1]}")
        print(f"  PCA values: {X_pca[i][:min(5, len(X_pca[i]))]}")
    
    print("\n" + "=" * 70)


def main():
    """
    Main execution function.
    """
    # File path
    filepath = "../../Datasets/pca_data.csv"
    
    print("\n[1] Loading and preparing data...")
    X_scaled, scaler = load_and_prepare_data(filepath)
    print(f"Data shape: {X_scaled.shape}")
    print(f"Features: {X_scaled.shape[1]}")
    print(f"Samples: {X_scaled.shape[0]}")
    
    print("\n[2] Determining optimal number of components...")
    n_components, pca_full = determine_optimal_components(X_scaled, variance_threshold=0.95)
    
    print(f"\n[3] Applying PCA with {n_components} components...")
    pca, X_pca = apply_pca(X_scaled, n_components)
    print(f"Transformation complete!")
    print(f"New shape: {X_pca.shape}")
    
    print("\n[4] Analyzing components...")
    analyze_components(pca)
    
    print("\n[5] Evaluating PCA...")
    metrics, X_reconstructed = evaluate_pca(pca, X_scaled, X_pca)
    
    print("\n[6] Displaying results...")
    display_results(pca, metrics, X_scaled, X_pca)
    
    # Save transformed data (optional)
    print("\n[7] Transformed Data Summary")
    print("-" * 70)
    print(f"Original size: {X_scaled.nbytes} bytes")
    print(f"Compressed size: {X_pca.nbytes} bytes")
    print(f"Size reduction: {(1 - X_pca.nbytes / X_scaled.nbytes):.1%}")
    
    print("\nProcess complete!")


if __name__ == "__main__":
    main()
