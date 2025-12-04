# 🌟 Diabetes Progression Prediction: Multiple Linear Regression
## Project Overview 📊
This repository contains a Machine Learning project where a **Multiple Linear Regression (MLR)** model is implemented using **NumPy**. The goal is to predict the quantitative measure of **diabetes disease progression (Y)** one year after baseline, based on 10 demographic and clinical features. The model is trained using the **Batch Gradient Descent** optimization algorithm, offering a deep insight into the core mathematics of linear models.

---

## 🔬 Core Machine Learning Technique: Multiple Linear Regression

Multiple Linear Regression is a supervised learning algorithm that assumes a linear relationship between a target variable ($Y$) and a set of independent features ($X_1, X_2, \ldots, X_n$).

### The Model Equation

The prediction ($\hat{y}$) is calculated as a linear combination of the features and a set of learned coefficients (parameters, $\mathbf{\theta}$):

$$
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n
$$

In **vectorized form**, which is used throughout the `code.py` implementation:

$$
\hat{\mathbf{y}} = X \mathbf{\theta}
$$

Where $X$ is the feature matrix augmented with a column of ones for the intercept $\theta_0$, and $\mathbf{\theta}$ is the vector of all coefficients.

### The Cost Function: Mean Squared Error (MSE)

To quantify the model's error, we use the **Mean Squared Error (MSE)**, which measures the average squared difference between the predicted values ($\hat{\mathbf{y}}$) and the true values ($\mathbf{y}$). Our objective is to find the parameter vector $\mathbf{\theta}$ that minimizes this cost function, $J(\mathbf{\theta})$:

$$
J(\mathbf{\theta}) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 = \frac{1}{2m} \|X \mathbf{\theta} - \mathbf{y}\|^2
$$

The $\frac{1}{2}$ term is included for mathematical convenience, as its derivative is simpler.

### Optimization: Batch Gradient Descent (BGD) 📉

**Batch Gradient Descent** is the engine that tunes the parameters $\mathbf{\theta}$ to minimize the MSE cost. It is an iterative process that works as follows:

1.  **Initialization:** Start with an initial guess for $\mathbf{\theta}$ (typically all zeros, as in the code).
2.  **Gradient Calculation:** The core of the algorithm involves calculating the **gradient** $\nabla J(\mathbf{\theta})$, which is the vector of partial derivatives, indicating the direction of steepest *ascent* on the cost surface. Because this is **Batch** Gradient Descent, the gradient is calculated using **all** $m$ training examples in a single pass (batch).
   
$$
\nabla J(\mathbf{\theta}) = \frac{1}{m} X^T (X \mathbf{\theta} - \mathbf{y})
$$
    
4.  **Parameter Update:** The parameters are updated by moving a small step in the direction *opposite* to the gradient (i.e., the direction of steepest **descent**). The step size is controlled by the **learning rate** ($\alpha$):

$$
\mathbf{\theta}_{new} = \mathbf{\theta}_{old} - \alpha \nabla J(\mathbf{\theta})
$$
    
6.  **Convergence:** This process is repeated for a set number of **iterations** (e.g., 2000), iteratively adjusting $\mathbf{\theta}$ until the cost function converges to a minimum.

---

## 🚀 Getting Started

### Prerequisites

You need a Python environment with the following libraries:
* `numpy` (for vectorized mathematical operations)
* `pandas` (for data loading and manipulation)
* `matplotlib` (for visualization)

Install them via pip:
```bash
pip install numpy pandas matplotlib
