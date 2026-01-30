# Basics of Machine Learning

A comprehensive, production-grade repository covering fundamental Machine Learning algorithms with educational implementations from scratch and using standard libraries.

## Repository Structure

This repository is organized into algorithm-specific modules, each containing:
- **README.md**: Detailed concept explanation, mathematical intuition, and use cases
- **Code/From_Scratch**: Pure Python implementations using only NumPy
- **Code/Using_Libraries**: Practical implementations using Scikit-learn
- **Datasets**: Sample datasets for experimentation
- **PPT**: Placeholder for presentation materials

## Algorithms Covered

### Supervised Learning
1. **Linear Regression** - Predicting continuous values
2. **Logistic Regression** - Binary and multiclass classification
3. **Decision Trees** - Tree-based classification and regression
4. **K-Nearest Neighbors (KNN)** - Instance-based learning
5. **Naive Bayes** - Probabilistic classification
6. **Support Vector Machines (SVM)** - Maximum margin classification

### Unsupervised Learning
1. **K-Means Clustering** - Partitioning data into clusters
2. **Principal Component Analysis (PCA)** - Dimensionality reduction

## Getting Started

### Prerequisites
```bash
Python 3.8+
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd "Basics Of ML"

# Install required packages
pip install numpy pandas scikit-learn matplotlib
```

### Usage
Each algorithm folder contains standalone code examples:
```bash
# Run from-scratch implementation
python "01_Linear_Regression/Code/From_Scratch/linear_regression_scratch.py"

# Run library-based implementation
python "01_Linear_Regression/Code/Using_Libraries/linear_regression_sklearn.py"
```

## Learning Path

**Beginners**: Start with Linear Regression, then move to Logistic Regression and KNN.

**Intermediate**: Explore Decision Trees, Naive Bayes, and K-Means Clustering.

**Advanced**: Deep dive into SVM, PCA, and implementation details.

## Tech Stack
- **Python**: Core programming language
- **NumPy**: Numerical computations for from-scratch implementations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Industry-standard ML library
- **Matplotlib**: Visualization (minimal usage)

## Contributing
This repository is designed for educational purposes. Contributions should maintain:
- Clean, well-commented code
- Beginner-friendly explanations
- Consistency with existing structure
- No unnecessary dependencies

## Project Goals
- Provide clear mathematical intuition for each algorithm
- Bridge theory and practice with dual implementations
- Offer production-ready code patterns
- Prepare learners for technical interviews
- Maintain reproducibility and modularity

## License
Educational use. Please provide attribution when using this material.

## Author
Sahana - 2026

## Acknowledgments
Built following best practices for educational ML repositories with focus on clarity, correctness, and learning value.
