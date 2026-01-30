# Decision Trees

## Concept Overview

Decision Trees are a non-parametric supervised learning algorithm used for both classification and regression tasks. They work by recursively partitioning the feature space into regions, creating a tree-like structure where each internal node represents a decision based on a feature, each branch represents the outcome of that decision, and each leaf node represents a class label or continuous value.

The algorithm learns a hierarchy of if-else questions that lead to predictions. It mimics human decision-making by breaking down complex decisions into simpler, sequential choices.

## Mathematical Intuition

### Information Gain (ID3, C4.5 algorithms)
Measures the reduction in entropy after a split:
```
Information Gain = Entropy(parent) - Σ(wᵢ × Entropy(childᵢ))
```

### Entropy (measure of impurity)
```
Entropy(S) = -Σ pᵢ log₂(pᵢ)
```
Where pᵢ is the proportion of class i in set S.
- Entropy = 0: Pure node (all samples same class)
- Entropy = 1: Maximum impurity (equal distribution)

### Gini Impurity (CART algorithm)
```
Gini(S) = 1 - Σ pᵢ²
```
Measures the probability of incorrectly classifying a randomly chosen element.
- Gini = 0: Pure node
- Gini = 0.5: Maximum impurity (binary classification)

### Split Criterion
For each feature and threshold:
```
Best split = argmax(Information Gain) or argmin(Gini Impurity)
```

### Regression Trees
Use Mean Squared Error (MSE) for splitting:
```
MSE = (1/n) Σ(yᵢ - ȳ)²
```

### Pruning
Cost-complexity pruning:
```
Cost = Error(T) + α × |Leaves(T)|
```
Where α controls the trade-off between tree size and accuracy.

## Algorithm Steps

1. **Start at Root**: Begin with all training samples at the root node
2. **Select Best Split**: For each feature and possible split point:
   - Calculate information gain or Gini impurity reduction
   - Choose the split that maximizes information gain (or minimizes impurity)
3. **Create Child Nodes**: Split data into subsets based on the chosen feature and threshold
4. **Recursive Partitioning**: Repeat steps 2-3 for each child node
5. **Stopping Criteria**: Stop when:
   - Node is pure (all samples same class)
   - Maximum depth reached
   - Minimum samples per node reached
   - No information gain possible
6. **Assign Predictions**: 
   - Classification: Majority class in leaf
   - Regression: Mean value in leaf
7. **Pruning (optional)**: Remove nodes that don't improve validation performance

## Real-World Use Cases

1. **Medical Diagnosis**: Disease diagnosis, treatment recommendation systems
2. **Finance**: Credit approval, loan default prediction, fraud detection
3. **Customer Analytics**: Customer segmentation, churn prediction
4. **Manufacturing**: Quality control, defect detection, process optimization
5. **HR**: Resume screening, employee attrition prediction
6. **Marketing**: Customer targeting, campaign effectiveness
7. **E-commerce**: Product recommendation, purchase prediction
8. **Telecommunications**: Network troubleshooting, service quality prediction
9. **Insurance**: Risk assessment, claim prediction
10. **Agriculture**: Crop disease detection, yield prediction

## Pros and Cons

### Advantages
- **Interpretable**: Easy to visualize and explain to non-technical stakeholders
- **No Feature Scaling Required**: Works with raw data, no normalization needed
- **Handles Mixed Data**: Can process both numerical and categorical features
- **Non-linear Relationships**: Captures complex non-linear patterns automatically
- **Feature Importance**: Automatically identifies important features
- **Missing Values**: Can handle missing data with surrogate splits
- **Fast Prediction**: O(log n) prediction time after training
- **No Assumptions**: Non-parametric, makes no distributional assumptions

### Disadvantages
- **Overfitting Prone**: Can create overly complex trees that memorize training data
- **High Variance**: Small changes in data can result in very different trees
- **Biased Toward Dominant Classes**: Imbalanced data leads to biased trees
- **Greedy Algorithm**: Locally optimal splits may not lead to globally optimal tree
- **Difficult to Capture Linear Relationships**: Requires many splits for simple linear patterns
- **Instability**: Small data variations cause large structural changes
- **Limited Extrapolation**: Cannot predict beyond training data range
- **Axis-aligned Splits**: Only creates rectangular decision boundaries

## Key Takeaways

1. Decision Trees partition feature space using recursive binary splits
2. Entropy and Gini impurity measure node purity
3. Information gain determines the best feature to split on
4. Pruning prevents overfitting by removing unnecessary branches
5. Maximum depth and minimum samples per leaf are key hyperparameters
6. Trees are interpretable but prone to high variance
7. Ensemble methods (Random Forest, Gradient Boosting) overcome single tree limitations
8. Feature importance scores identify influential features
9. Trees handle non-linear relationships and interactions naturally
10. No feature scaling required, making them convenient for mixed data types

## Common Interview Questions

**Q1: What is the difference between Gini Impurity and Entropy?**
A: Both measure node impurity. Entropy uses logarithm (-Σpᵢlog₂pᵢ) and ranges 0-1, while Gini uses sum of squared probabilities (1-Σpᵢ²) and is computationally faster. Entropy penalizes impure nodes more heavily. In practice, they often produce similar trees. Gini is preferred in CART, entropy in ID3/C4.5.

**Q2: How do you prevent overfitting in Decision Trees?**
A: Methods include: (1) Set maximum depth, (2) Require minimum samples per leaf/split, (3) Pruning (remove nodes that don't improve validation performance), (4) Limit maximum leaf nodes, (5) Set minimum impurity decrease threshold, (6) Use ensemble methods like Random Forest.

**Q3: What is pruning and why is it important?**
A: Pruning removes branches that have little predictive power, reducing overfitting. Pre-pruning stops growth early using stopping criteria. Post-pruning (cost-complexity pruning) builds full tree then removes nodes that don't improve validation accuracy. It improves generalization and reduces model complexity.

**Q4: How do Decision Trees handle missing values?**
A: Methods include: (1) Surrogate splits - find alternative features that give similar splits, (2) Separate category for missing values, (3) Imputation before training, (4) Weighted splits considering only non-missing values. Scikit-learn doesn't natively handle missing values; preprocessing is required.

**Q5: What is information gain and how is it calculated?**
A: Information gain measures entropy reduction from a split: IG = Entropy(parent) - weighted average of Entropy(children). It quantifies how much uncertainty is reduced by splitting on a feature. Higher information gain means better split. Used in ID3 and C4.5 algorithms.

**Q6: Explain the difference between ID3, C4.5, and CART algorithms.**
A: ID3 uses information gain (biased toward features with many values), only handles categorical features. C4.5 uses gain ratio (normalized information gain), handles continuous features and missing values, includes pruning. CART uses Gini impurity, creates binary trees, supports regression, used in scikit-learn.

**Q7: Why are Decision Trees called greedy algorithms?**
A: At each step, trees choose the best split for the current node without considering future splits. This locally optimal approach doesn't guarantee a globally optimal tree. The tree might make a suboptimal split early that prevents finding better overall structure later.

**Q8: How do Decision Trees handle categorical vs numerical features?**
A: Numerical features: Test all unique values as thresholds (feature <= threshold). Categorical features (unordered): Test all possible subsets of categories. Categorical features (ordered): Treat as numerical. Most implementations require one-hot encoding for categorical features.

**Q9: What is the difference between Decision Tree and Random Forest?**
A: A Decision Tree is a single tree that may overfit. Random Forest is an ensemble of multiple trees trained on random subsets of data and features. Random Forest reduces variance through averaging, improves accuracy, and is more robust but less interpretable than a single tree.

**Q10: Can Decision Trees handle multi-output problems?**
A: Yes, CART-based implementations can handle multi-output regression and classification. Each leaf stores multiple target values. For classification, it predicts probability distributions for each output. For regression, it predicts mean values for each target variable.
