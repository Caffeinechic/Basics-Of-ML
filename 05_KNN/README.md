# K-Nearest Neighbors (KNN)

## Concept Overview

K-Nearest Neighbors is a simple, non-parametric, lazy learning algorithm used for both classification and regression. It makes predictions by finding the K closest training examples in the feature space and using their labels to predict the label of a new data point. For classification, it uses majority voting; for regression, it uses averaging.

KNN stores all training data and makes decisions based on the entire training set during prediction, making it instance-based rather than model-based.

## Mathematical Intuition

### Distance Metrics

**Euclidean Distance (most common):**
```
d(x, y) = √(Σ(xᵢ - yᵢ)²)
```

**Manhattan Distance:**
```
d(x, y) = Σ|xᵢ - yᵢ|
```

**Minkowski Distance (generalization):**
```
d(x, y) = (Σ|xᵢ - yᵢ|ᵖ)^(1/p)
```
Where p=1 gives Manhattan, p=2 gives Euclidean.

### Classification
```
ŷ = mode(labels of K nearest neighbors)
```
Predict the most common class among K neighbors.

### Regression
```
ŷ = (1/K) Σ(yᵢ for i in K nearest neighbors)
```
Predict the average value of K neighbors.

### Weighted KNN
Closer neighbors get more influence:
```
weight = 1 / distance
ŷ = Σ(weight × label) / Σ(weight)
```

## Algorithm Steps

1. **Store Training Data**: Keep all training examples (no explicit training phase)
2. **Receive New Data Point**: Get features of the point to classify/predict
3. **Calculate Distances**: Compute distance from new point to all training points
4. **Find K Nearest**: Select K training examples with smallest distances
5. **Aggregate Labels**:
   - Classification: Majority vote among K neighbors
   - Regression: Mean value of K neighbors
6. **Return Prediction**: Output the aggregated result

## Real-World Use Cases

1. **Recommendation Systems**: Product recommendations based on similar user preferences
2. **Image Recognition**: Handwriting recognition, face detection
3. **Medical Diagnosis**: Disease prediction based on patient similarity
4. **Credit Rating**: Assess creditworthiness based on similar profiles
5. **Pattern Recognition**: Identify patterns in customer behavior
6. **Anomaly Detection**: Detect outliers based on distance to neighbors
7. **Missing Value Imputation**: Fill missing data with neighbor averages
8. **Spell Checkers**: Suggest corrections based on similar words
9. **Stock Market Prediction**: Predict trends based on historical patterns
10. **Biometrics**: Fingerprint and signature verification

## Pros and Cons

### Advantages
- **Simple to Understand**: Intuitive concept, easy to explain
- **No Training Phase**: Just stores data, fast setup
- **Non-parametric**: Makes no assumptions about data distribution
- **Naturally Handles Multi-class**: Works with any number of classes
- **Adapts to New Data**: Easy to add new training examples
- **Effective for Small Datasets**: Works well with limited data
- **Versatile**: Can be used for classification, regression, imputation

### Disadvantages
- **Computationally Expensive**: O(n) prediction time, slow for large datasets
- **Memory Intensive**: Stores entire training dataset
- **Curse of Dimensionality**: Performance degrades with many features
- **Sensitive to Irrelevant Features**: All features treated equally
- **Requires Feature Scaling**: Distance metrics sensitive to scale
- **Sensitive to Noisy Data**: Outliers significantly affect predictions
- **Choosing K is Critical**: Wrong K leads to poor performance
- **No Model Interpretability**: Cannot explain feature importance

## Key Takeaways

1. KNN is a lazy learner - no training, all work done during prediction
2. Choice of K is crucial - small K sensitive to noise, large K oversmooths
3. Distance metric matters - Euclidean is standard but not always best
4. Feature scaling is mandatory for fair distance calculations
5. Odd K values prevent ties in binary classification
6. Cross-validation helps determine optimal K
7. Performance degrades in high dimensions (curse of dimensionality)
8. Weighted KNN gives more influence to closer neighbors
9. Time complexity: Training O(1), Prediction O(nd) where n=samples, d=dimensions
10. Can be optimized using KD-trees or Ball trees for faster neighbor search

## Common Interview Questions

**Q1: How do you choose the optimal value of K?**
A: Use cross-validation to test different K values. Plot accuracy vs K (validation curve). Small K (1-3) causes high variance and overfitting. Large K causes high bias and underfitting. Odd K avoids ties in binary classification. Typically K = sqrt(n) is a reasonable starting point. Common range: 3-10.

**Q2: What is the curse of dimensionality in KNN?**
A: In high-dimensional spaces, all points become approximately equidistant, making nearest neighbor concepts meaningless. Distance measures lose discriminative power. Solution: dimensionality reduction (PCA), feature selection, or use algorithms designed for high dimensions.

**Q3: Why is feature scaling important for KNN?**
A: KNN uses distance metrics which are sensitive to feature scales. Features with larger ranges dominate the distance calculation. Example: age (0-100) vs salary (0-100000) - salary would dominate. Use standardization (z-score) or normalization (min-max scaling) to give equal importance to all features.

**Q4: What is the difference between KNN and K-Means?**
A: KNN is supervised classification/regression that predicts labels based on K nearest training examples. K-Means is unsupervised clustering that partitions data into K clusters. KNN needs labeled data, K-Means doesn't. KNN has no training phase, K-Means iteratively updates centroids.

**Q5: How can you speed up KNN for large datasets?**
A: Use efficient data structures: (1) KD-Tree - works well in low dimensions, (2) Ball Tree - better for higher dimensions, (3) Approximate nearest neighbors (ANN) - trade accuracy for speed, (4) Locality-Sensitive Hashing (LSH), (5) Reduce dataset size with clustering, (6) Dimensionality reduction.

**Q6: When would you use weighted KNN instead of standard KNN?**
A: Weighted KNN when: (1) Neighbors at different distances should have different influence, (2) Dealing with imbalanced data, (3) K is large but you want closer neighbors to matter more. Weights typically: 1/distance or 1/distance². Prevents distant neighbors from equally influencing predictions.

**Q7: How does KNN handle imbalanced datasets?**
A: Standard KNN biased toward majority class. Solutions: (1) Use weighted KNN giving more weight to minority class, (2) Adjust K value, (3) Use distance-weighted voting, (4) Oversample minority or undersample majority class, (5) Use stratified sampling, (6) Consider other algorithms like SMOTE-edited KNN.

**Q8: What distance metric should you use for categorical features?**
A: Euclidean doesn't work for categorical data. Use: (1) Hamming distance - counts mismatches, (2) Overlap metric - similarity coefficient, (3) Heterogeneous distance metrics combining numerical and categorical, (4) One-hot encoding then Euclidean (creates high dimensions), (5) Gower distance for mixed data types.

**Q9: Explain the bias-variance tradeoff in choosing K.**
A: Small K (e.g., K=1): Low bias, high variance - overfits, sensitive to noise, irregular decision boundaries. Large K: High bias, low variance - underfits, smoother boundaries, less sensitive to noise but may miss patterns. Optimal K balances this tradeoff, typically found via cross-validation.

**Q10: Can KNN be used for online learning?**
A: Yes and no. KNN naturally adapts to new data by simply adding new points. However, it doesn't scale well for streaming data due to: (1) Growing memory requirements, (2) Increasing prediction time, (3) No model compression. Solutions: maintain a sliding window, use approximate methods, or consider incremental learning algorithms.
