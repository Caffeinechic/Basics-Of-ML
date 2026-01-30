# Logistic Regression

## Concept Overview

Logistic Regression is a supervised learning algorithm used for binary and multiclass classification problems. Despite its name, it is a classification algorithm, not a regression algorithm. It predicts the probability that an instance belongs to a particular class using a logistic (sigmoid) function.

The algorithm models the relationship between independent variables and a categorical dependent variable by estimating probabilities using a logistic function, which maps any real-valued number to a value between 0 and 1.

## Mathematical Intuition

### Sigmoid Function
The core of logistic regression is the sigmoid (logistic) function:
```
σ(z) = 1 / (1 + e^(-z))
```
Where:
- z = linear combination of features = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
- Output range: (0, 1)
- Output interpretation: probability of class 1

### Prediction Function
```
ŷ = σ(wᵀx + b)
```
Where:
- w = weight vector
- x = feature vector
- b = bias term

### Decision Boundary
Classification decision:
```
ŷ = 1 if σ(z) >= 0.5
ŷ = 0 if σ(z) < 0.5
```

### Cost Function (Log Loss / Binary Cross-Entropy)
```
J(w, b) = -(1/n) Σ[yᵢ log(ŷᵢ) + (1 - yᵢ) log(1 - ŷᵢ)]
```
Where:
- n = number of samples
- yᵢ = true label (0 or 1)
- ŷᵢ = predicted probability

### Gradient Descent Update
```
w := w - α × (1/n) Σ[(ŷᵢ - yᵢ) × xᵢ]
b := b - α × (1/n) Σ(ŷᵢ - yᵢ)
```
Where α is the learning rate.

### Multiclass Extension (Softmax)
For K classes:
```
P(y = k | x) = exp(zₖ) / Σexp(zⱼ)
```

## Algorithm Steps

1. **Initialize**: Set random weights and bias to small values
2. **Compute Linear Combination**: Calculate z = wᵀx + b
3. **Apply Sigmoid**: Compute σ(z) to get probabilities
4. **Calculate Loss**: Compute binary cross-entropy loss
5. **Compute Gradients**: Calculate derivatives of loss with respect to parameters
6. **Update Parameters**: Apply gradient descent to update weights and bias
7. **Iterate**: Repeat steps 2-6 until convergence or max iterations
8. **Predict**: Apply threshold to probabilities for classification

## Real-World Use Cases

1. **Medical Diagnosis**: Disease prediction (diabetes, cancer), patient risk assessment
2. **Finance**: Credit approval, fraud detection, default prediction
3. **Marketing**: Customer churn prediction, email spam classification
4. **E-commerce**: Click-through rate prediction, conversion prediction
5. **Social Media**: Sentiment analysis, fake news detection
6. **Human Resources**: Employee attrition prediction, resume screening
7. **Insurance**: Claim approval, risk assessment
8. **Transportation**: Accident prediction, traffic violation detection
9. **Telecommunications**: Network fault detection, customer retention
10. **Manufacturing**: Quality control, defect detection

## Pros and Cons

### Advantages
- **Probabilistic Output**: Provides probability scores, not just class labels
- **Simple and Interpretable**: Easy to understand and implement
- **Computationally Efficient**: Fast training and prediction
- **No Feature Scaling Required**: Works without normalization (though recommended)
- **Low Variance**: Less prone to overfitting with regularization
- **Works Well**: Effective for linearly separable data
- **Multiclass Support**: Can be extended using One-vs-Rest or softmax
- **Online Learning**: Can update model with new data incrementally

### Disadvantages
- **Assumes Linearity**: Decision boundary is linear, cannot capture complex patterns
- **Sensitive to Outliers**: Extreme values can distort the decision boundary
- **Requires Large Sample Size**: Needs sufficient data for reliable estimates
- **Feature Engineering Needed**: Non-linear relationships require manual feature creation
- **Multicollinearity Issues**: Correlated features affect coefficient interpretation
- **Imbalanced Data Sensitivity**: Biased toward majority class without adjustment
- **Not Suitable for Complex Problems**: Limited modeling capacity for intricate patterns
- **Independence Assumption**: Assumes features are independent

## Key Takeaways

1. Logistic Regression is for classification, not regression tasks
2. Sigmoid function maps linear combinations to probabilities (0-1)
3. Binary Cross-Entropy is the standard loss function
4. Decision boundary is linear in feature space
5. Regularization (L1/L2) prevents overfitting and improves generalization
6. Feature scaling improves convergence speed
7. Probability calibration ensures reliable confidence scores
8. Use evaluation metrics appropriate for classification (accuracy, precision, recall, F1, AUC-ROC)
9. Handle imbalanced datasets using class weights or resampling
10. Logistic Regression is a baseline model for binary classification

## Common Interview Questions

**Q1: Why is it called Logistic Regression when it's used for classification?**
A: The name comes from the logistic (sigmoid) function used to model probabilities. It performs regression on log-odds (logit) of the probability, hence the name. However, by applying a threshold to the predicted probabilities, it performs classification.

**Q2: What is the difference between Linear and Logistic Regression?**
A: Linear Regression predicts continuous values using a linear function (output: -∞ to +∞). Logistic Regression predicts probabilities using the sigmoid function (output: 0 to 1) for classification tasks. Linear uses MSE loss, Logistic uses log loss.

**Q3: Explain the sigmoid function and why it's used.**
A: The sigmoid function σ(z) = 1/(1 + e^(-z)) maps any real number to a value between 0 and 1, making it perfect for probability estimation. It has an S-shaped curve, is differentiable (needed for gradient descent), and has a clear probabilistic interpretation.

**Q4: What is the log loss (binary cross-entropy) and why use it?**
A: Log loss penalizes confident incorrect predictions heavily. It's defined as -[y log(ŷ) + (1-y) log(1-ŷ)]. It's convex, which ensures gradient descent finds the global minimum, and it's the maximum likelihood estimate for binary classification.

**Q5: How do you handle multiclass classification with Logistic Regression?**
A: Two approaches: (1) One-vs-Rest (OvR): Train K binary classifiers, each distinguishing one class from all others. (2) Multinomial (Softmax): Directly model probabilities for all K classes simultaneously using softmax function.

**Q6: What is regularization in Logistic Regression? L1 vs L2?**
A: Regularization prevents overfitting by adding a penalty term. L1 (Lasso) adds |w| penalty, leading to sparse models (feature selection). L2 (Ridge) adds w² penalty, shrinking coefficients but keeping all features. Elastic Net combines both.

**Q7: How do you evaluate a Logistic Regression model?**
A: Use classification metrics: Accuracy (overall correctness), Precision (positive prediction reliability), Recall (sensitivity), F1-score (harmonic mean of precision/recall), AUC-ROC (discrimination ability), Confusion Matrix (detailed breakdown), and Log Loss (probability quality).

**Q8: How do you handle imbalanced datasets in Logistic Regression?**
A: Methods include: (1) Class weights - penalize misclassifying minority class more, (2) Resampling - oversample minority or undersample majority, (3) Threshold adjustment - lower threshold for minority class, (4) SMOTE - synthetic minority oversampling, (5) Anomaly detection approaches.

**Q9: What assumptions does Logistic Regression make?**
A: Key assumptions: (1) Linear relationship between log-odds and features, (2) Independence of observations, (3) Little to no multicollinearity among features, (4) Large sample size, (5) Binary or ordinal dependent variable.

**Q10: What is the difference between logistic regression and SVM?**
A: Logistic Regression uses log loss and predicts probabilities, works best when data is linearly separable with some noise. SVM uses hinge loss, finds maximum-margin separator, and can handle non-linear boundaries with kernels. SVM is less sensitive to outliers and doesn't provide probabilities directly.
