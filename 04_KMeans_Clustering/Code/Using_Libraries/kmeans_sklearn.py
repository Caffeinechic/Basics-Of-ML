"""
K-Means Clustering Implementation using Scikit-learn
Production-ready implementation with evaluation
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(filepath):
    """
    Load and prepare data for clustering.
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        X_scaled: Scaled features
        scaler: Fitted scaler object
    """
    # Load data
    data = pd.read_csv(filepath)
    X = data.values
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler


def find_optimal_k(X, k_range=(2, 10)):
    """
    Find optimal number of clusters using multiple methods.
    
    Args:
        X: Scaled data
        k_range: Tuple of (min_k, max_k)
    
    Returns:
        Dictionary with evaluation metrics for each K
    """
    results = {
        'k_values': [],
        'inertia': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': []
    }
    
    print("\n[Elbow Method Analysis]")
    print("-" * 70)
    print(f"{'K':<5} {'Inertia':<12} {'Silhouette':<12} {'Davies-Bouldin':<16} {'Calinski-Harabasz':<18}")
    print("-" * 70)
    
    for k in range(k_range[0], k_range[1]):
        model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = model.fit_predict(X)
        
        inertia = model.inertia_
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        
        results['k_values'].append(k)
        results['inertia'].append(inertia)
        results['silhouette'].append(silhouette)
        results['davies_bouldin'].append(davies_bouldin)
        results['calinski_harabasz'].append(calinski_harabasz)
        
        print(f"{k:<5} {inertia:<12.2f} {silhouette:<12.4f} {davies_bouldin:<16.4f} {calinski_harabasz:<18.2f}")
    
    return results


def train_model(X, n_clusters=3):
    """
    Train K-Means model.
    
    Args:
        X: Scaled features
        n_clusters: Number of clusters
    
    Returns:
        Trained model and cluster labels
    """
    model = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42
    )
    labels = model.fit_predict(X)
    return model, labels


def evaluate_clustering(X, model, labels):
    """
    Evaluate clustering quality.
    
    Args:
        X: Scaled features
        model: Trained model
        labels: Cluster labels
    
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'inertia': model.inertia_,
        'silhouette': silhouette_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'n_iter': model.n_iter_
    }
    return metrics


def display_results(model, metrics, labels, X, feature_names=None):
    """
    Display comprehensive clustering results.
    
    Args:
        model: Trained model
        metrics: Evaluation metrics
        labels: Cluster labels
        X: Original scaled data
        feature_names: List of feature names (optional)
    """
    print("=" * 70)
    print("K-Means Clustering with Scikit-learn - Results")
    print("=" * 70)
    
    # Model info
    print("\n[1] Model Information")
    print("-" * 70)
    print(f"Number of clusters: {model.n_clusters}")
    print(f"Number of iterations: {metrics['n_iter']}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    
    # Quality metrics
    print("\n[2] Clustering Quality Metrics")
    print("-" * 70)
    print(f"Inertia (WCSS): {metrics['inertia']:.2f}")
    print(f"  Lower is better, measures cluster compactness")
    
    print(f"\nSilhouette Score: {metrics['silhouette']:.4f}")
    if metrics['silhouette'] > 0.5:
        print("  Interpretation: Strong, well-separated clusters")
    elif metrics['silhouette'] > 0.25:
        print("  Interpretation: Moderate cluster structure")
    else:
        print("  Interpretation: Weak or overlapping clusters")
    
    print(f"\nDavies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
    print("  Lower is better, measures cluster separation")
    
    print(f"\nCalinski-Harabasz Score: {metrics['calinski_harabasz']:.2f}")
    print("  Higher is better, ratio of between/within cluster variance")
    
    # Cluster distribution
    print("\n[3] Cluster Distribution")
    print("-" * 70)
    for i in range(model.n_clusters):
        count = np.sum(labels == i)
        percentage = (count / len(labels)) * 100
        print(f"Cluster {i}: {count:>5} samples ({percentage:>5.1f}%)")
    
    # Cluster centroids
    print("\n[4] Cluster Centroids (scaled)")
    print("-" * 70)
    if feature_names:
        print(f"{'Cluster':<10}", end="")
        for name in feature_names:
            print(f"{name:<12}", end="")
        print()
        print("-" * 70)
        for i, centroid in enumerate(model.cluster_centers_):
            print(f"{i:<10}", end="")
            for val in centroid:
                print(f"{val:<12.4f}", end="")
            print()
    else:
        for i, centroid in enumerate(model.cluster_centers_):
            print(f"Cluster {i}: {centroid}")
    
    # Cluster statistics
    print("\n[5] Cluster Statistics")
    print("-" * 70)
    for i in range(model.n_clusters):
        cluster_data = X[labels == i]
        print(f"\nCluster {i}:")
        print(f"  Size: {len(cluster_data)}")
        print(f"  Mean distance to centroid: {np.mean(np.sqrt(np.sum((cluster_data - model.cluster_centers_[i])**2, axis=1))):.4f}")
        print(f"  Max distance to centroid: {np.max(np.sqrt(np.sum((cluster_data - model.cluster_centers_[i])**2, axis=1))):.4f}")
    
    # Sample assignments
    print("\n[6] Sample Cluster Assignments (First 10 samples)")
    print("-" * 70)
    print(f"{'Sample':<10} {'Cluster':<10} {'Distance to Centroid':<25}")
    print("-" * 70)
    for i in range(min(10, len(labels))):
        distance = np.sqrt(np.sum((X[i] - model.cluster_centers_[labels[i]])**2))
        print(f"{i + 1:<10} {labels[i]:<10} {distance:<25.4f}")
    
    print("\n" + "=" * 70)


def predict_new_data(model, scaler, new_data):
    """
    Predict cluster for new data.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        new_data: New data points
    
    Returns:
        Cluster labels and distances to centroids
    """
    new_data = np.array(new_data).reshape(-1, model.n_features_in_)
    new_data_scaled = scaler.transform(new_data)
    predictions = model.predict(new_data_scaled)
    distances = model.transform(new_data_scaled)
    return predictions, distances


def main():
    """
    Main execution function.
    """
    # File path
    filepath = "../../Datasets/clustering_data.csv"
    
    print("\n[1] Loading and preparing data...")
    X_scaled, scaler = load_and_prepare_data(filepath)
    print(f"Number of samples: {X_scaled.shape[0]}")
    print(f"Number of features: {X_scaled.shape[1]}")
    
    print("\n[2] Finding optimal number of clusters...")
    optimal_k_results = find_optimal_k(X_scaled, k_range=(2, 8))
    
    # Recommend optimal K based on silhouette score
    best_k_idx = np.argmax(optimal_k_results['silhouette'])
    optimal_k = optimal_k_results['k_values'][best_k_idx]
    print(f"\nRecommended K based on Silhouette Score: {optimal_k}")
    
    print(f"\n[3] Training final model with K={optimal_k}...")
    model, labels = train_model(X_scaled, n_clusters=optimal_k)
    print("Model training complete!")
    
    print("\n[4] Evaluating clustering...")
    metrics = evaluate_clustering(X_scaled, model, labels)
    
    print("\n[5] Displaying results...")
    display_results(model, metrics, labels, X_scaled)
    
    # Example: Predicting on new data
    print("\n[7] Example: Predicting cluster for new data")
    print("-" * 70)
    new_data = X_scaled[:3]  # Using first 3 samples as example
    predictions, distances = predict_new_data(model, scaler, new_data)
    print(f"Predictions: {predictions}")
    print(f"Distances to all centroids:\n{distances}")
    
    print("\nProcess complete!")


if __name__ == "__main__":
    main()
