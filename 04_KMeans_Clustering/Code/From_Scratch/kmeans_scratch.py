"""
K-Means Clustering Implementation from Scratch
Using only NumPy for educational purposes
"""

import numpy as np
import pandas as pd


class KMeansScratch:
    """
    K-Means Clustering implemented from scratch.
    
    Attributes:
        n_clusters: Number of clusters (K)
        max_iters: Maximum number of iterations
        random_state: Random seed for reproducibility
        centroids: Cluster centroids
        labels: Cluster assignments for training data
        inertia: Sum of squared distances to closest centroid
    """
    
    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        """
        Initialize K-Means model.
        
        Args:
            n_clusters: Number of clusters (default: 3)
            max_iters: Maximum iterations (default: 100)
            random_state: Random seed (default: None)
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None
    
    def fit(self, X):
        """
        Fit K-Means model to data.
        
        Args:
            X: Data, shape (n_samples, n_features)
        """
        X = np.array(X)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Initialize centroids randomly
        random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_indices].copy()
        
        # Iterative optimization
        for iteration in range(self.max_iters):
            # Store old centroids to check convergence
            old_centroids = self.centroids.copy()
            
            # Assignment step: assign points to nearest centroid
            self.labels = self._assign_clusters(X)
            
            # Update step: recalculate centroids
            self._update_centroids(X)
            
            # Check convergence
            if np.allclose(old_centroids, self.centroids):
                print(f"Converged at iteration {iteration + 1}")
                break
            
            if (iteration + 1) % 10 == 0:
                inertia = self._calculate_inertia(X)
                print(f"Iteration {iteration + 1}/{self.max_iters}, Inertia: {inertia:.2f}")
        
        # Calculate final inertia
        self.inertia = self._calculate_inertia(X)
    
    def _assign_clusters(self, X):
        """
        Assign each point to nearest centroid.
        
        Args:
            X: Data
        
        Returns:
            Cluster labels
        """
        distances = self._calculate_distances(X, self.centroids)
        return np.argmin(distances, axis=1)
    
    def _calculate_distances(self, X, centroids):
        """
        Calculate Euclidean distances between points and centroids.
        
        Args:
            X: Data points
            centroids: Cluster centroids
        
        Returns:
            Distance matrix (n_samples, n_clusters)
        """
        distances = np.zeros((len(X), len(centroids)))
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        return distances
    
    def _update_centroids(self, X):
        """
        Update centroids as mean of assigned points.
        
        Args:
            X: Data
        """
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                self.centroids[i] = cluster_points.mean(axis=0)
    
    def _calculate_inertia(self, X):
        """
        Calculate within-cluster sum of squares (WCSS).
        
        Args:
            X: Data
        
        Returns:
            Inertia value
        """
        inertia = 0
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[i]) ** 2)
        return inertia
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Args:
            X: New data points
        
        Returns:
            Cluster labels
        """
        X = np.array(X)
        distances = self._calculate_distances(X, self.centroids)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X):
        """
        Fit model and return cluster labels.
        
        Args:
            X: Data
        
        Returns:
            Cluster labels
        """
        self.fit(X)
        return self.labels


def calculate_silhouette_score(X, labels):
    """
    Calculate simplified silhouette score.
    
    Args:
        X: Data
        labels: Cluster labels
    
    Returns:
        Average silhouette score
    """
    n_samples = len(X)
    silhouette_scores = []
    
    for i in range(n_samples):
        # Points in same cluster
        same_cluster = X[labels == labels[i]]
        if len(same_cluster) <= 1:
            continue
        
        # Mean distance to points in same cluster (a)
        a = np.mean(np.sqrt(np.sum((same_cluster - X[i]) ** 2, axis=1)))
        
        # Mean distance to points in nearest other cluster (b)
        b = float('inf')
        for cluster_label in np.unique(labels):
            if cluster_label != labels[i]:
                other_cluster = X[labels == cluster_label]
                mean_dist = np.mean(np.sqrt(np.sum((other_cluster - X[i]) ** 2, axis=1)))
                b = min(b, mean_dist)
        
        # Silhouette coefficient
        s = (b - a) / max(a, b) if max(a, b) > 0 else 0
        silhouette_scores.append(s)
    
    return np.mean(silhouette_scores) if silhouette_scores else 0


def main():
    """
    Demonstration of K-Means from scratch.
    """
    print("=" * 60)
    print("K-Means Clustering from Scratch - Demo")
    print("=" * 60)
    
    # Load dataset
    print("\n[1] Loading dataset...")
    data = pd.read_csv("../../Datasets/clustering_data.csv")
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {data.columns.tolist()}")
    
    # Prepare data
    X = data.values
    
    # Feature scaling (standardization)
    print("\n[2] Scaling features...")
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / X_std
    
    # Train model with different K values (Elbow method)
    print("\n[3] Finding optimal K using Elbow method...")
    inertias = []
    k_range = range(2, 8)
    
    for k in k_range:
        model = KMeansScratch(n_clusters=k, max_iters=100, random_state=42)
        model.fit(X_scaled)
        inertias.append(model.inertia)
        print(f"K={k}, Inertia={model.inertia:.2f}")
    
    # Train final model with optimal K
    optimal_k = 3
    print(f"\n[4] Training final model with K={optimal_k}...")
    final_model = KMeansScratch(n_clusters=optimal_k, max_iters=100, random_state=42)
    labels = final_model.fit_predict(X_scaled)
    
    # Evaluate clustering
    print("\n[5] Clustering Results")
    print("-" * 60)
    print(f"Number of clusters: {optimal_k}")
    print(f"Inertia (WCSS): {final_model.inertia:.2f}")
    
    # Cluster sizes
    print("\nCluster sizes:")
    for i in range(optimal_k):
        count = np.sum(labels == i)
        percentage = (count / len(labels)) * 100
        print(f"  Cluster {i}: {count} samples ({percentage:.1f}%)")
    
    # Silhouette score
    silhouette = calculate_silhouette_score(X_scaled, labels)
    print(f"\nSilhouette Score: {silhouette:.4f}")
    if silhouette > 0.5:
        print("  Interpretation: Strong cluster structure")
    elif silhouette > 0.25:
        print("  Interpretation: Moderate cluster structure")
    else:
        print("  Interpretation: Weak cluster structure")
    
    # Centroids
    print("\n[6] Cluster Centroids (scaled)")
    print("-" * 60)
    for i, centroid in enumerate(final_model.centroids):
        print(f"Cluster {i}: {centroid}")
    
    # Sample assignments
    print("\n[7] Sample Cluster Assignments (First 10 samples)")
    print("-" * 60)
    print(f"{'Sample':<10} {'Cluster':<10}")
    print("-" * 60)
    for i in range(min(10, len(labels))):
        print(f"{i + 1:<10} {labels[i]:<10}")
    
    print("\n" + "=" * 60)
    print("Clustering Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
