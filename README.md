# Multiple Linear, Ridge, and Lasso Regression on the Iris Dataset

A practical implementation and visual walkthrough comparing **Ordinary Least Squares (Multiple Linear Regression)** with **L2 (Ridge)** and **L1 (Lasso)** regularization using `scikit-learn` on the classic Iris dataset.

![Iris Dataset Overview](iris%20data%20info%20image%20multi,ridge.jpeg)

## 📌 Overview

Regularization helps prevent overfitting and manage multicollinearity by adding a penalty term to the regression loss function:

* **Multiple Linear Regression (OLS):** Fits a linear hyperplane without coefficient penalties.
* **Ridge Regression (L2 Penalty):** Shrinks correlated feature coefficients smoothly toward zero without eliminating them.
* **Lasso Regression (L1 Penalty):** Performs automatic feature selection by driving less informative feature coefficients to exact zero.

## 📂 Repository Contents

| File | Description |
| :--- | :--- |
| [`multiple_ridge_lasso_iris.py`](./multiple_ridge_lasso_iris.py) | End-to-end Python script implementing data preparation, model training, regularization comparison, and evaluation metrics. |
| [`Multiple_Ridge_Lasso_Regression_Sklearn.pdf`](./Multiple_Ridge_Lasso_Regression_Sklearn.pdf) | Step-by-step visual guide and mathematical breakdown of Multiple, Ridge, and Lasso regression in `scikit-learn`. |
| [`iris data info image multi,ridge.jpeg`](./iris%20data%20info%20image%20multi,ridge.jpeg) | Summary visual of the Iris dataset features and structure. |

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Libraries:** `scikit-learn`, `numpy`, `pandas`, `matplotlib`

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mdanwarulkarim/Multiple-Linear-Regression-Ridge-Lesso-Algorithm.git](https://github.com/mdanwarulkarim/Multiple-Linear-Regression-Ridge-Lesso-Algorithm.git)
   cd Multiple-Linear-Regression-Ridge-Lesso-Algorithm
