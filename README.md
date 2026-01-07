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
2.  **Gradient Calculation:** The core of the algorithm involves calculating the **gradient** $\nabla J(\mathbf{\theta})$, which is the vector of partial derivatives, indicating the direction of steepest _ascent_ on the cost surface. Because this is **Batch** Gradient Descent, the gradient is calculated using **all** $m$ training examples in a single pass (batch).

$$
\nabla J(\mathbf{\theta}) = \frac{1}{m} X^T (X \mathbf{\theta} - \mathbf{y})
$$

4.  **Parameter Update:** The parameters are updated by moving a small step in the direction _opposite_ to the gradient (i.e., the direction of steepest **descent**). The step size is controlled by the **learning rate** ($\alpha$):

$$
\mathbf{\theta}_{new} = \mathbf{\theta}_{old} - \alpha \nabla J(\mathbf{\theta})
$$

6.  **Convergence:** This process is repeated for a set number of **iterations** (2000), iteratively adjusting $\mathbf{\theta}$ until the cost function converges to a minimum.

---

## 📝 Code Explanation

The `code.py` script implements the model through the following key steps:

### 1. Load Dataset
The data is loaded from `assets/diabetes.tab.csv`. The script separates the features (`X`) from the target variable (`y`), converting `y` into a 2D numpy array to comply with matrix operation standards.

The dataset contains **10 input features (X1 to X10)**:
*   **X1 (Age)**: Age of the patient.
*   **X2 (Gender)**: Gender of the patient.
*   **X3 (BMI)**: Body Mass Index.
*   **X4 (BP)**: Average blood pressure.
*   **X5 (S1)**: Total serum cholesterol (TC).
*   **X6 (S2)**: Low‑density lipoproteins, LDL cholesterol.
*   **X7 (S3)**: High‑density lipoproteins, HDL cholesterol.
*   **X8 (S4)**: Total cholesterol / HDL ratio (TCH).
*   **X9 (S5)**: Log of serum triglycerides level (often noted as LTG).
*   **X10 (S6)**: Blood sugar (plasma glucose) level (often noted as GLU).

### 2. Normalization
The input features are normalized to improve gradient descent performance:
- **Mean Centering**: Subtracting the mean from each column.
- **Scaling**: Dividing by the range (max - min).
- **Intercept**: A column of ones is added to `X` to account for the bias term ($\theta_0$).

### 3. Hyperparameters (Optimized)
The training is controlled by pre-set hyperparameters:
- **Learning Rate (`alpha`):** `0.01`
- **Iterations:** `2000`
- **m:** The total number of training examples.

### 4. Initialize Theta
The parameter vector `theta` is initialized as a vector of zeros, with dimensions corresponding to the number of features (including the intercept).

### 5. Gradient Descent
The weights are updated iteratively to minimize the error:
1. **Prediction**: Compute hypothesis $h_\theta(x) = X\theta$.
2. **Error**: Calculate difference between predictions and actual values.
3. **Gradient**: Compute the gradient of the cost function.
4. **Update**: Adjust theta using $\theta = \theta - \alpha \cdot \text{gradient}$.
5. **Cost**: Store the Mean Squared Error (MSE) to monitor convergence.

### 6. Final Predictions
After the loop finishes, the optimized `theta` is used to generate the final predictions. The code outputs the learned parameters and generates a visualization comparing predicted vs. actual values.

---

## 🚀 Getting Started

### Requirements

- Python = 3.13.3
- NumPy = 2.2.6
- Pandas = 2.2.3
- Matplotlib = 3.10.3

### Installation

### 1) Install python 3.13.3

```bash
$ sudo apt update
$ sudo apt install python3.13.3
```

### 2) Install the required packages

```bash
$ pip install -r requirements.txt
```

### Results

The result of the model (results/performance.png) is shown in the following figure:

![Results](results/Performance.png)
