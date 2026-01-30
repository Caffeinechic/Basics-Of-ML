"""
Principal Component Analysis (PCA) Implementation from Scratch
Using only NumPy - implements eigenvalue decomposition for dimensionality reduction
"""

import numpy as np
import pandas as pd


class PCAScratch:
    """
    PCA implemented from scratch using eigenvalue decomposition.
    
    Attributes:
        n_components: Number of components to keep
        components: Principal components (eigenvectors)
        explained_variance: Variance explained by each component
        explained_variance_ratio: Proportion of variance explained
        mean: Mean of training data
        eigenvalues: Eigenvalues from decomposition
    """
    
    def __init__(self, n_components=None):
        """
        Initialize PCA.
        
        Args:
            n_components: Number of components to keep (default: all)
        """
        self.n_components = n_components
        self.components = np.array([[]])
        self.explained_variance = np.array([])
        self.explained_variance_ratio = np.array([])
        self.mean = np.array([])
        self.eigenvalues = np.array([])
    
    def fit(self, X):
        """
        Fit PCA model by computing principal components.
        
        Args:
            X: Training data, shape (n_samples, n_features)
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Set default n_components
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)
        
        # Step 1: Center the data (subtract mean)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Step 2: Compute covariance matrix
        # Cov = (1/(n-1)) * X^T * X
        cov_matrix = np.cov(X_centered.T)
        
        # Step 3: Compute eigenvalues and eigenvectors
        # eigenvalues = variance along each principal component
        # eigenvectors = directions of principal components
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Step 4: Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Step 5: Store top n_components
        self.eigenvalues = eigenvalues[:self.n_components]
        self.components = eigenvectors[:, :self.n_components].T
        
        # Step 6: Calculate explained variance
        self.explained_variance = self.eigenvalues
        
        # Step 7: Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = self.eigenvalues / total_variance
        
        print(f"Fitted PCA with {self.n_components} components")
        print(f"Explained variance ratio: {self.explained_variance_ratio}")
        print(f"Cumulative explained variance: {np.cumsum(self.explained_variance_ratio)}")
    
    def transform(self, X):
        """
        Transform data to principal component space.
        
        Args:
            X: Data to transform, shape (n_samples, n_features)
        
        Returns:
            Transformed data, shape (n_samples, n_components)
        """
        X = np.array(X)
        
        # Center the data
        X_centered = X - self.mean
        
        # Project onto principal components
        # Z = X * W where W is the component matrix
        if self.components.size == 0:
            return np.array([])
        X_transformed = np.dot(X_centered, self.components.T)
        
        return X_transformed
    
    def fit_transform(self, X):
        """
        Fit PCA and transform data in one step.
        
        Args:
            X: Training data
        
        Returns:
            Transformed data
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        Reconstruct original data from principal components.
        
        Args:
            X_transformed: Transformed data, shape (n_samples, n_components)
        
        Returns:
            Reconstructed data, shape (n_samples, n_features)
        """
        X_transformed = np.array(X_transformed)
        
        # Reconstruct: X_reconstructed = Z * W^T + mean
        if self.components.size == 0:
            return np.array([])
        X_reconstructed = np.dot(X_transformed, self.components) + self.mean
        
        return X_reconstructed
    
    def get_covariance(self):
        """
        Get the estimated covariance of the data.
        
        Returns:
            Covariance matrix
        """
        # Cov = W * diag(explained_variance) * W^T
        if self.components.size == 0 or self.explained_variance.size == 0:
            return np.array([[]])
        return np.dot(self.components.T * self.explained_variance, self.components)


def main():
    """
    Demonstration of PCA from scratch.
    """
    print("=" * 70)
    print("Principal Component Analysis from Scratch - Demo")
    print("=" * 70)
    
    # Load dataset
    print("\n[1] Loading dataset...")
    data = pd.read_csv("../../Datasets/pca_data.csv")
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {data.columns.tolist()}")
    
    # Prepare data
    X = data.values
    
    # Standardize data (important for PCA)
    print("\n[2] Standardizing features...")
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_standardized = (X - X_mean) / X_std
    
    print(f"Original shape: {X.shape}")
    
    # Determine optimal number of components
    print("\n[3] Analyzing explained variance...")
    pca_full = PCAScratch(n_components=None)
    pca_full.fit(X_standardized)
    
    if pca_full.explained_variance_ratio.size == 0:
        print("Error: No components found")
        return
    
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio)
    print("\nExplained Variance by Component:")
    for i, (var, cum_var) in enumerate(zip(pca_full.explained_variance_ratio, cumulative_variance)):
        print(f"  PC{i + 1}: {var:.4f} (Cumulative: {cum_var:.4f})")
    
    # Choose number of components for 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"\nComponents needed for 95% variance: {n_components_95}")
    
    # Apply PCA with selected components
    print(f"\n[4] Applying PCA with {n_components_95} components...")
    pca = PCAScratch(n_components=n_components_95)
    X_pca = pca.fit_transform(X_standardized)
    
    print(f"Reduced shape: {X_pca.shape}")
    print(f"Dimensionality reduction: {X.shape[1]} -> {X_pca.shape[1]}")
    if pca.explained_variance_ratio.size > 0:
        print(f"Retained variance: {np.sum(pca.explained_variance_ratio):.4f}")
    
    # Display principal components
    print("\n[5] Principal Components (Loadings)")
    print("-" * 70)
    if pca.components.size > 0:
        for i, component in enumerate(pca.components):
            print(f"PC{i + 1}: {component}")
    
    # Reconstruction error
    print("\n[6] Reconstruction Analysis")
    print("-" * 70)
    X_reconstructed = pca.inverse_transform(X_pca)
    reconstruction_error = np.mean((X_standardized - X_reconstructed) ** 2)
    print(f"Mean Squared Reconstruction Error: {reconstruction_error:.6f}")
    
    # Sample transformation
    print("\n[7] Sample Transformations (First 5 samples)")
    print("-" * 70)
    print("Original data (standardized) -> PCA space:")
    for i in range(min(5, len(X_pca))):
        print(f"Sample {i + 1}:")
        print(f"  Original: {X_standardized[i][:3]}... ({len(X_standardized[i])} features)")
        print(f"  PCA: {X_pca[i]} ({len(X_pca[i])} components)")
    
    # Variance statistics
    print("\n[8] Variance Statistics")
    print("-" * 70)
    if pca_full.eigenvalues.size > 0 and pca.explained_variance.size > 0:
        print(f"Total variance in original data: {np.sum(pca_full.eigenvalues):.4f}")
        print(f"Variance retained: {np.sum(pca.explained_variance):.4f}")
        print(f"Variance lost: {np.sum(pca_full.eigenvalues) - np.sum(pca.explained_variance):.4f}")
    print(f"Compression ratio: {X_pca.shape[1] / X.shape[1]:.2%}")
    
    # Component importance
    print("\n[9] Component Importance")
    print("-" * 70)
    if pca.explained_variance_ratio.size > 0:
        for i in range(len(pca.explained_variance_ratio)):
            bar_length = int(pca.explained_variance_ratio[i] * 50)
            bar = '#' * bar_length
            print(f"PC{i + 1}: {bar} {pca.explained_variance_ratio[i]:.4f}")
    
    print("\n" + "=" * 70)
    print("PCA Analysis Complete!")
    print("=" * 70)
    print(f"\nKey Insights:")
    print(f"- Reduced from {X.shape[1]} to {X_pca.shape[1]} dimensions")
    if pca.explained_variance_ratio.size > 0:
        print(f"- Retained {np.sum(pca.explained_variance_ratio):.1%} of variance")
        print(f"- First PC explains {pca.explained_variance_ratio[0]:.1%} of variance")


if __name__ == "__main__":
    main()
