# Linear Regression

## Concept Overview

Linear Regression is a supervised learning algorithm used to model the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to observed data. It is one of the simplest and most widely used algorithms in machine learning and statistics.

The goal is to find the best-fitting straight line (or hyperplane in multiple dimensions) through the data points that minimizes the prediction error.

## Mathematical Intuition

### Simple Linear Regression (One Feature)
The equation of a line:
```
y = mx + b
```
In ML notation:
```
y = β₀ + β₁x
```
Where:
- y = predicted value (dependent variable)
- x = input feature (independent variable)
- β₀ = intercept (bias term)
- β₁ = slope (weight/coefficient)

### Multiple Linear Regression (Multiple Features)
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```
Matrix form:
```
y = Xβ + ε
```
Where:
- X = feature matrix (n samples × m features)
- β = coefficient vector
- ε = error term

### Cost Function (Mean Squared Error)
```
J(β) = (1/2n) Σ(yᵢ - ŷᵢ)²
```
Where:
- n = number of samples
- yᵢ = actual value
- ŷᵢ = predicted value

### Optimization (Gradient Descent)
Update rule:
```
β := β - α × ∂J(β)/∂β
```
Where α is the learning rate.

### Closed-Form Solution (Normal Equation)
```
β = (XᵀX)⁻¹Xᵀy
```

## Algorithm Steps

1. **Initialize**: Set random weights (β) and bias (β₀)
2. **Forward Pass**: Calculate predictions using current parameters
3. **Calculate Loss**: Compute Mean Squared Error
4. **Compute Gradients**: Calculate partial derivatives of loss with respect to parameters
5. **Update Parameters**: Adjust weights using gradient descent
6. **Iterate**: Repeat steps 2-5 until convergence or max iterations
7. **Predict**: Use learned parameters for new data

## Real-World Use Cases

1. **Price Prediction**: Real estate prices based on size, location, amenities
2. **Sales Forecasting**: Predict sales based on advertising spend, seasonality
3. **Medical Research**: Estimate disease progression based on patient metrics
4. **Finance**: Stock price prediction, risk assessment
5. **Agriculture**: Crop yield prediction based on weather, soil conditions
6. **Marketing**: Customer lifetime value prediction
7. **Economics**: GDP forecasting, demand estimation
8. **Environmental Science**: Temperature prediction, pollution level estimation

## Pros and Cons

### Advantages
- **Simple and Interpretable**: Easy to understand and explain to stakeholders
- **Fast Training**: Computationally efficient, even with large datasets
- **Low Variance**: Less prone to overfitting with sufficient data
- **Closed-Form Solution**: Can be solved directly without iteration
- **Works Well**: Effective when relationship is approximately linear
- **No Hyperparameter Tuning**: Minimal configuration required
- **Baseline Model**: Good starting point for regression problems

### Disadvantages
- **Assumes Linearity**: Poor performance when true relationship is non-linear
- **Sensitive to Outliers**: Extreme values significantly affect the fit
- **Multicollinearity Issues**: Correlated features cause unstable estimates
- **Limited Complexity**: Cannot capture complex patterns
- **Requires Feature Engineering**: Non-linear relationships need manual feature creation
- **Homoscedasticity Assumption**: Assumes constant variance in errors
- **Independence Assumption**: Assumes independent observations

## Key Takeaways

1. Linear Regression models linear relationships between features and target
2. Mean Squared Error is the standard loss function for regression
3. Gradient Descent and Normal Equation are two optimization approaches
4. Feature scaling improves convergence speed in gradient descent
5. R² score measures the proportion of variance explained by the model
6. Residual analysis helps validate model assumptions
7. Regularization (Ridge/Lasso) prevents overfitting in high-dimensional data
8. Always check for linearity, independence, and homoscedasticity assumptions

## Common Interview Questions

**Q1: What is the difference between Simple and Multiple Linear Regression?**
A: Simple Linear Regression uses one independent variable, while Multiple Linear Regression uses two or more independent variables to predict the target.

**Q2: Explain the difference between Gradient Descent and Normal Equation.**
A: Normal Equation computes optimal parameters directly using matrix operations (XᵀX)⁻¹Xᵀy, while Gradient Descent iteratively updates parameters. Normal Equation is exact but slow for large feature sets (O(n³)), while Gradient Descent scales better but requires hyperparameter tuning.

**Q3: How do you handle outliers in Linear Regression?**
A: Methods include: removing outliers using IQR or z-scores, using robust regression techniques (RANSAC, Huber loss), applying log transformation to target, or using regularization to reduce their impact.

**Q4: What is R² score and how is it calculated?**
A: R² (coefficient of determination) measures the proportion of variance in the dependent variable explained by independent variables. R² = 1 - (SS_res / SS_tot), where SS_res is residual sum of squares and SS_tot is total sum of squares. Range: 0 to 1 (higher is better).

**Q5: What assumptions does Linear Regression make?**
A: Key assumptions are: (1) Linearity between features and target, (2) Independence of observations, (3) Homoscedasticity (constant variance of errors), (4) Normal distribution of errors, (5) No multicollinearity among features.

**Q6: When should you use Ridge vs Lasso regression?**
A: Ridge (L2) is preferred when all features are relevant but need shrinkage, as it shrinks coefficients but keeps all features. Lasso (L1) is better for feature selection as it can shrink coefficients to exactly zero, effectively removing features.

**Q7: How does feature scaling affect Linear Regression?**
A: Feature scaling is not required for the Normal Equation but is crucial for Gradient Descent. Without scaling, features with larger ranges dominate the gradient, causing slow convergence and requiring smaller learning rates.

**Q8: What is the difference between correlation and regression?**
A: Correlation measures the strength and direction of linear relationship between two variables (symmetric). Regression models the relationship to predict one variable from another (asymmetric) and provides the equation of the relationship.
