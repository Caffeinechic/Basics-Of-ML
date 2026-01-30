# Principal Component Analysis (PCA)

## Concept Overview

Principal Component Analysis is an unsupervised learning technique used for dimensionality reduction. It transforms high-dimensional data into a lower-dimensional space while preserving as much variance (information) as possible. PCA finds new orthogonal axes (principal components) that capture the maximum variance in the data.

The first principal component captures the most variance, the second captures the second most (orthogonal to the first), and so on. This allows us to reduce features while retaining the most important information.

## Mathematical Intuition

### Objective
Find directions of maximum variance in the data.

### Steps

**1. Standardize data:**
```
X_std = (X - μ) / σ
```

**2. Compute covariance matrix:**
```
Cov(X) = (1/(n-1)) X^T X
```

**3. Eigenvalue decomposition:**
```
Cov(X) = V Λ V^T
```
Where:
- V = eigenvectors (principal components)
- Λ = diagonal matrix of eigenvalues (variance explained)

**4. Sort eigenvalues:**
Order eigenvectors by eigenvalue magnitude (descending).

**5. Select components:**
Keep top k eigenvectors forming matrix W (d × k).

**6. Transform data:**
```
Z = X W
```
Where Z is the reduced representation.

### Variance Explained
```
Explained variance ratio = λᵢ / Σλⱼ
```

### Reconstruction
```
X_reconstructed = Z W^T + μ
```

## Algorithm Steps

1. **Standardize Data**: Center data (mean=0) and scale (std=1)
2. **Compute Covariance Matrix**: Calculate pairwise covariances between features
3. **Calculate Eigenvalues and Eigenvectors**: Decompose covariance matrix
4. **Sort Eigenvalues**: Order from largest to smallest
5. **Select k Components**: Choose number of components to retain
6. **Form Projection Matrix**: Use top k eigenvectors
7. **Transform Data**: Project original data onto principal components
8. **Evaluate**: Check cumulative explained variance

## Real-World Use Cases

1. **Data Visualization**: Reduce to 2D/3D for plotting high-dimensional data
2. **Image Compression**: Reduce image dimensions while preserving quality
3. **Noise Reduction**: Remove noise by keeping high-variance components
4. **Feature Extraction**: Create new features for machine learning
5. **Preprocessing**: Speed up learning algorithms by reducing dimensions
6. **Genetics**: Gene expression analysis, population studies
7. **Face Recognition**: Eigenfaces for facial recognition systems
8. **Finance**: Portfolio optimization, risk analysis
9. **Anomaly Detection**: Identify outliers in reduced space
10. **Recommender Systems**: Collaborative filtering dimensionality reduction

## Pros and Cons

### Advantages
- **Reduces Dimensionality**: Decreases computational cost and storage
- **Removes Multicollinearity**: Creates uncorrelated features
- **Noise Reduction**: Filters out low-variance noise
- **Visualization**: Enables plotting of high-dimensional data
- **Improves Performance**: Can improve model accuracy by removing noise
- **No Tuning Required**: Deterministic, no hyperparameters (except k)
- **Fast Computation**: Efficient for moderate-sized datasets
- **Interpretable Components**: Each component is linear combination of features

### Disadvantages
- **Linear Transformation Only**: Cannot capture non-linear relationships
- **Loses Interpretability**: Principal components hard to interpret
- **Sensitive to Scale**: Requires feature standardization
- **Information Loss**: Discarding components loses some information
- **Assumes High Variance = Important**: Not always true
- **Outlier Sensitive**: Outliers can distort principal components
- **Not Sparse**: All features contribute to each component
- **Direction, Not Magnitude**: Doesn't consider prediction target

## Key Takeaways

1. PCA finds orthogonal directions of maximum variance
2. Principal components are uncorrelated by construction
3. First PC captures most variance, second captures second most, etc.
4. Feature standardization is mandatory before PCA
5. Choose k components to retain 90-95% cumulative variance
6. PCA is unsupervised - doesn't use target labels
7. Eigenvectors are the principal components (directions)
8. Eigenvalues represent variance explained by each component
9. Use scree plot or cumulative variance to select k
10. PCA assumes linear relationships in data

## Common Interview Questions

**Q1: What is the difference between PCA and feature selection?**
A: Feature selection chooses a subset of original features, keeping them interpretable. PCA creates new features (principal components) that are linear combinations of all original features, losing direct interpretability but capturing variance more efficiently. PCA: transformation. Feature selection: selection.

**Q2: How do you choose the number of principal components?**
A: Methods: (1) Cumulative explained variance - keep components until reaching threshold (e.g., 95%), (2) Scree plot - look for elbow where eigenvalues plateau, (3) Kaiser criterion - keep components with eigenvalue > 1, (4) Cross-validation - choose k that gives best downstream model performance, (5) Domain knowledge - based on application needs.

**Q3: Why is feature scaling important for PCA?**
A: PCA is based on variance. Features with larger scales (e.g., income vs age) will dominate the principal components even if less important. Standardization (mean=0, std=1) ensures all features contribute equally based on their variance structure, not their measurement scale.

**Q4: What is the relationship between PCA and SVD (Singular Value Decomposition)?**
A: PCA can be computed using SVD. For centered data X, SVD gives X = UΣV^T. PCA eigenvectors are columns of V, and eigenvalues are Σ²/(n-1). SVD is numerically more stable and efficient than computing covariance matrix explicitly. Scikit-learn's PCA uses SVD internally.

**Q5: Can PCA be used for non-linear data?**
A: Standard PCA is linear. For non-linear relationships, use Kernel PCA which applies kernel trick (similar to SVM) to perform PCA in high-dimensional feature space, capturing non-linear patterns. Other alternatives: t-SNE, UMAP, Autoencoders for non-linear dimensionality reduction.

**Q6: What is explained variance ratio?**
A: Explained variance ratio is the proportion of total variance explained by each principal component. Formula: λᵢ/Σλⱼ where λᵢ is the eigenvalue of component i. Sum of all ratios = 1. It quantifies how much information each component captures. Use cumulative sum to decide how many components to keep.

**Q7: How does PCA handle outliers?**
A: PCA is sensitive to outliers because it maximizes variance, and outliers have high variance. Outliers can distort principal components. Solutions: (1) Remove outliers before PCA, (2) Use robust PCA variants, (3) Winsorize data, (4) Use other dimensionality reduction methods less sensitive to outliers (e.g., ICA).

**Q8: What is the difference between PCA and LDA (Linear Discriminant Analysis)?**
A: PCA is unsupervised, maximizes variance, doesn't use class labels. LDA is supervised, maximizes class separability (between-class variance / within-class variance), uses class labels. PCA for dimensionality reduction, LDA for classification preprocessing. LDA requires labeled data and can produce at most (classes-1) components.

**Q9: Can you reverse PCA (reconstruct original data)?**
A: Yes, approximately. X_reconstructed = Z W^T + μ, where Z is reduced data, W^T is transpose of component matrix, μ is original mean. If using all components, reconstruction is exact. If using k < d components, some information is lost, causing reconstruction error proportional to discarded variance.

**Q10: When should you NOT use PCA?**
A: Avoid PCA when: (1) Features are already uncorrelated, (2) All features are important and interpretable (e.g., medical diagnosis), (3) Non-linear relationships exist (use Kernel PCA instead), (4) Data has many outliers, (5) Sparse data (text, collaborative filtering - preserving sparsity is important), (6) Small number of features (overhead not worth it), (7) Features have inherent meaning that must be preserved.
