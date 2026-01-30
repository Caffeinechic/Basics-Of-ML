# K-Means Clustering

## Concept Overview

K-Means is an unsupervised learning algorithm used for clustering data into K distinct groups. It partitions n observations into K clusters where each observation belongs to the cluster with the nearest mean (centroid). The algorithm aims to minimize the variance within each cluster.

Unlike supervised learning, K-Means works with unlabeled data to discover hidden patterns and natural groupings based on feature similarity.

## Mathematical Intuition

### Objective Function
Minimize the Within-Cluster Sum of Squares (WCSS):
```
J = Σ(i=1 to K) Σ(x∈Cᵢ) ||x - μᵢ||²
```
Where:
- K = number of clusters
- Cᵢ = cluster i
- x = data point
- μᵢ = centroid of cluster i
- ||x - μᵢ||² = squared Euclidean distance

### Distance Metric (Euclidean)
```
d(x, μ) = √(Σ(xⱼ - μⱼ)²)
```

### Centroid Update
```
μᵢ = (1/|Cᵢ|) Σ(x∈Cᵢ) x
```
Mean of all points assigned to cluster i.

### Convergence
Algorithm converges when:
- Centroids no longer move
- Assignments no longer change
- Maximum iterations reached

## Algorithm Steps

1. **Initialize**: Randomly select K data points as initial centroids
2. **Assignment Step**: Assign each data point to the nearest centroid
   - Calculate distance from each point to all centroids
   - Assign point to cluster with minimum distance
3. **Update Step**: Recalculate centroids as mean of all points in each cluster
4. **Check Convergence**: 
   - If centroids haven't changed or max iterations reached, stop
   - Otherwise, repeat steps 2-3
5. **Output**: Final cluster assignments and centroids

## Real-World Use Cases

1. **Customer Segmentation**: Group customers by behavior, demographics, purchase patterns
2. **Image Compression**: Reduce colors by clustering similar pixel values
3. **Document Clustering**: Group similar documents, articles, or emails
4. **Anomaly Detection**: Identify outliers that don't fit any cluster
5. **Market Segmentation**: Identify distinct market segments for targeted marketing
6. **Social Network Analysis**: Discover communities in networks
7. **Gene Sequence Analysis**: Group similar gene expression patterns
8. **City Planning**: Identify optimal locations for facilities based on demographics
9. **Inventory Management**: Group products with similar demand patterns
10. **Recommendation Systems**: Cluster users or items for collaborative filtering

## Pros and Cons

### Advantages
- **Simple and Fast**: Easy to implement and computationally efficient O(nKI)
- **Scalable**: Works well with large datasets
- **Guaranteed Convergence**: Always converges to local optimum
- **Easy Interpretation**: Cluster centers provide clear interpretation
- **Works Well**: Effective when clusters are spherical and well-separated
- **Adaptable**: Can be modified for different distance metrics
- **Popular**: Widely used with extensive library support

### Disadvantages
- **Requires K**: Must specify number of clusters in advance
- **Sensitive to Initialization**: Different initializations yield different results
- **Local Optima**: Converges to local minimum, not global
- **Assumes Spherical Clusters**: Poor performance with non-spherical shapes
- **Sensitive to Outliers**: Outliers distort centroid positions
- **Scale Sensitive**: Features with larger ranges dominate distance calculations
- **Equal Cluster Size Bias**: Tends to create equal-sized clusters
- **No Probability**: Hard assignment, no uncertainty quantification

## Key Takeaways

1. K-Means partitions data into K clusters by minimizing within-cluster variance
2. Algorithm alternates between assignment and update steps
3. Euclidean distance is the standard metric (others possible)
4. Multiple runs with different initializations recommended
5. K-Means++ initialization improves convergence and results
6. Elbow method and silhouette score help determine optimal K
7. Feature scaling is crucial for fair distance calculations
8. Works best with spherical, well-separated, equal-variance clusters
9. Sensitive to outliers and initialization
10. Hard clustering (each point belongs to exactly one cluster)

## Common Interview Questions

**Q1: How do you choose the optimal number of clusters K?**
A: Methods include: (1) Elbow Method - plot WCSS vs K, look for elbow point where decrease slows, (2) Silhouette Score - measures cluster cohesion and separation, (3) Gap Statistic - compares WCSS to random data, (4) Domain knowledge - use business context, (5) Hierarchical clustering dendrogram for visual inspection.

**Q2: What is K-Means++ and why is it better than random initialization?**
A: K-Means++ is a smart initialization method that spreads initial centroids far apart. It selects the first centroid randomly, then each subsequent centroid is chosen with probability proportional to its squared distance from nearest existing centroid. This reduces sensitivity to initialization and typically leads to better, faster convergence than random initialization.

**Q3: What is the time complexity of K-Means?**
A: O(n × K × I × d), where n = number of samples, K = number of clusters, I = number of iterations, d = number of dimensions. Typically I and K are small, so it's approximately O(n) for fixed K and d, making it very scalable.

**Q4: How do you handle outliers in K-Means?**
A: Approaches include: (1) Remove outliers before clustering using IQR or z-scores, (2) Use K-Medoids (PAM) which uses medians instead of means, (3) Use DBSCAN for density-based clustering, (4) Apply robust scaling, (5) Use mixture models that assign soft probabilities.

**Q5: What is the difference between K-Means and K-Medoids?**
A: K-Means uses mean of points as centroids (can be anywhere), sensitive to outliers. K-Medoids uses actual data points as cluster centers (medoids), more robust to outliers and noise. K-Medoids is more expensive computationally O(n²) but better for non-Euclidean distances.

**Q6: Explain the elbow method for choosing K.**
A: Plot the WCSS (inertia) against number of clusters K. As K increases, WCSS decreases. The elbow point where the rate of decrease sharply slows is the optimal K. It represents the point where adding more clusters doesn't significantly improve fit. Note: elbow may not always be clear.

**Q7: What is inertia in K-Means?**
A: Inertia is the sum of squared distances of samples to their closest cluster center (WCSS). It measures cluster compactness. Lower inertia means tighter clusters. However, inertia always decreases with more clusters, so it must be balanced with cluster count (hence elbow method).

**Q8: How does K-Means handle categorical data?**
A: Standard K-Means doesn't handle categorical data well since Euclidean distance isn't meaningful. Solutions: (1) One-hot encoding (creates high dimensions), (2) K-Modes - uses mode instead of mean and Hamming distance, (3) K-Prototypes - combines K-Means and K-Modes for mixed data, (4) Convert categories to meaningful numerical values if ordinal.

**Q9: What is the difference between K-Means and hierarchical clustering?**
A: K-Means: requires K upfront, fast O(n), produces flat clusters, sensitive to initialization. Hierarchical: doesn't require K, slower O(n²) or O(n³), produces dendrogram showing cluster hierarchy, deterministic. Hierarchical better for exploring data structure, K-Means better for large datasets.

**Q10: Can K-Means find clusters of different sizes and densities?**
A: No, K-Means assumes clusters are spherical with similar variance and tends toward equal-sized clusters. For varying sizes/densities, use: DBSCAN (density-based), Gaussian Mixture Models (different variances), Mean Shift (adaptive bandwidth), or Spectral Clustering (non-convex shapes).
