# Support Vector Machines (SVM)

## Concept Overview

Support Vector Machines are powerful supervised learning algorithms used for classification and regression. SVM finds the optimal hyperplane that maximally separates different classes in the feature space. The key idea is to maximize the margin (distance) between the decision boundary and the closest data points from each class, called support vectors.

SVM can handle both linear and non-linear classification through the use of kernel functions, making it versatile for complex datasets.

## Mathematical Intuition

### Linear SVM

**Objective:** Maximize the margin between classes

**Hyperplane equation:**
```
w · x + b = 0
```
Where w is the weight vector and b is the bias.

**Margin:**
```
margin = 2 / ||w||
```

**Optimization problem:**
```
Minimize: (1/2)||w||²
Subject to: yᵢ(w · xᵢ + b) >= 1 for all i
```

### Soft Margin SVM (with slack variables)
```
Minimize: (1/2)||w||² + C Σξᵢ
Subject to: yᵢ(w · xᵢ + b) >= 1 - ξᵢ, ξᵢ >= 0
```
Where C is the regularization parameter and ξᵢ are slack variables allowing misclassification.

### Kernel Trick

Map data to higher dimensions using kernel function:
```
K(x, x') = φ(x) · φ(x')
```

**Common kernels:**
- Linear: K(x, x') = x · x'
- Polynomial: K(x, x') = (γx · x' + r)^d
- RBF (Gaussian): K(x, x') = exp(-γ||x - x'||²)
- Sigmoid: K(x, x') = tanh(γx · x' + r)

### Decision Function
```
f(x) = sign(Σ αᵢyᵢK(xᵢ, x) + b)
```
Where αᵢ are Lagrange multipliers from dual optimization.

## Algorithm Steps

1. **Input Data**: Receive training samples with labels
2. **Choose Kernel**: Select appropriate kernel function (linear, RBF, etc.)
3. **Set Parameters**: Choose C (regularization) and kernel parameters
4. **Solve Optimization**: Find optimal hyperplane by solving:
   - Maximize: Σαᵢ - (1/2)ΣΣαᵢαⱼyᵢyⱼK(xᵢ, xⱼ)
   - Subject to: 0 <= αᵢ <= C and Σαᵢyᵢ = 0
5. **Identify Support Vectors**: Points where αᵢ > 0
6. **Calculate Bias**: Using support vectors
7. **Prediction**: Use decision function with support vectors

## Real-World Use Cases

1. **Image Classification**: Face detection, object recognition
2. **Text Categorization**: Document classification, sentiment analysis
3. **Bioinformatics**: Protein classification, gene expression analysis
4. **Handwriting Recognition**: Digit and character recognition
5. **Medical Diagnosis**: Cancer detection, disease classification
6. **Financial Forecasting**: Stock market prediction, credit scoring
7. **Remote Sensing**: Land use classification, change detection
8. **Intrusion Detection**: Network security, anomaly detection
9. **Speech Recognition**: Phoneme classification
10. **Chemoinformatics**: Drug discovery, molecular classification

## Pros and Cons

### Advantages
- **Effective in High Dimensions**: Works well even when features > samples
- **Memory Efficient**: Uses subset of training points (support vectors)
- **Versatile**: Different kernel functions for various decision boundaries
- **Strong Theoretical Foundation**: Based on statistical learning theory
- **Robust to Outliers**: Less sensitive compared to other algorithms
- **Works Well with Clear Margin**: Excellent for well-separated classes
- **No Local Minimum**: Convex optimization problem
- **Good Generalization**: Maximizing margin reduces overfitting

### Disadvantages
- **Slow Training**: O(n²) to O(n³) complexity for large datasets
- **Memory Intensive**: Requires storing kernel matrix
- **Difficult to Interpret**: Black box model, hard to explain
- **Requires Feature Scaling**: Sensitive to feature scales
- **Sensitive to Parameters**: C and kernel parameters need careful tuning
- **No Probability Estimates**: Requires additional calibration
- **Poor Performance on Large Datasets**: Doesn't scale well
- **Choosing Kernel is Challenging**: Wrong kernel leads to poor results

## Key Takeaways

1. SVM finds optimal separating hyperplane by maximizing margin
2. Support vectors are the critical data points near decision boundary
3. Kernel trick enables non-linear classification without explicit mapping
4. C parameter controls trade-off between margin and misclassification
5. RBF kernel is most popular for non-linear problems
6. Feature scaling is mandatory for SVM
7. Works best with clear margin of separation
8. Not suitable for very large datasets due to computational complexity
9. Requires careful hyperparameter tuning (C, gamma for RBF)
10. One-vs-One or One-vs-Rest for multiclass classification

## Common Interview Questions

**Q1: What are support vectors and why are they important?**
A: Support vectors are the data points closest to the decision boundary (hyperplane) that lie on the margin boundaries. They are the most difficult to classify and define the position of the hyperplane. Only support vectors influence the model; removing other points doesn't change the hyperplane. This makes SVM memory-efficient.

**Q2: Explain the role of the C parameter in SVM.**
A: C is the regularization parameter controlling the trade-off between maximizing margin and minimizing classification error. Large C: small margin, fewer misclassifications (may overfit). Small C: large margin, more misclassifications (may underfit). It determines the penalty for misclassified points. Typical values: 0.1 to 100.

**Q3: What is the kernel trick and why is it useful?**
A: The kernel trick maps data to higher-dimensional space without explicitly computing the transformation. Instead of computing φ(x)·φ(x'), we compute K(x,x') directly. This enables SVM to find non-linear decision boundaries efficiently. It avoids the computational cost and curse of dimensionality from explicit high-dimensional mapping.

**Q4: When should you use linear kernel vs RBF kernel?**
A: Use Linear kernel when: (1) Data is linearly separable, (2) Many features (text classification), (3) Fast training needed. Use RBF kernel when: (1) Non-linear relationship, (2) Fewer features, (3) No prior knowledge of data structure. RBF is default choice when unsure. Always try linear first for high-dimensional data.

**Q5: What is the difference between hard margin and soft margin SVM?**
A: Hard margin SVM requires all points to be correctly classified with no violations (only for perfectly linearly separable data). Soft margin SVM allows some misclassifications using slack variables ξᵢ, controlled by C parameter. Soft margin is more practical as real data is rarely perfectly separable and may contain outliers.

**Q6: How does SVM handle multi-class classification?**
A: SVM is inherently binary. For multiclass: (1) One-vs-Rest (OvR): Train K binary classifiers, each separating one class from all others. Predict class with highest confidence. (2) One-vs-One (OvO): Train K(K-1)/2 classifiers for each pair of classes. Use voting to determine final class. OvO is more accurate but slower.

**Q7: What is the difference between SVM and logistic regression?**
A: SVM maximizes margin (distance to nearest points), focuses on support vectors, uses hinge loss. Logistic regression maximizes likelihood, uses all points, uses log loss. SVM better for clear margins and fewer samples. Logistic regression provides probabilities and is faster for large datasets. SVM with linear kernel and logistic regression often give similar results.

**Q8: Explain the gamma parameter in RBF kernel.**
A: Gamma (γ) defines the influence radius of support vectors. High γ: each point influences small radius, complex/wiggly decision boundary, prone to overfitting. Low γ: wider influence, smoother decision boundary, may underfit. γ = 1/(n_features × variance) is common default. Must tune with C via grid search.

**Q9: Why is feature scaling important for SVM?**
A: SVM uses distance metrics (in kernel functions). Features with larger scales dominate the distance calculation, biasing the optimization. Example: age (0-100) vs income (0-100000) - income would dominate. Standardization (mean=0, std=1) or normalization (0-1 range) ensures all features contribute equally.

**Q10: What are the computational challenges of SVM?**
A: Main challenges: (1) Training time O(n²) to O(n³) - slow for large datasets, (2) Memory requirement O(n²) for kernel matrix, (3) Hyperparameter tuning requires multiple training runs, (4) Prediction time O(n_support_vectors × n_features) - can be slow. Solutions: Use linear SVM for large data, subsample data, use approximate methods like LinearSVC, or switch to algorithms like logistic regression or gradient boosting.
