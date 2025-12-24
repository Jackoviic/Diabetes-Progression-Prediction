import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('assets/diabetes.tab.csv')
X = df.drop('Y', axis=1).values  # axis=1 >> drop column, .values >> convert to a 1D numpy array.
y = df[['Y']].to_numpy()  # Select as DataFrame column and convert to NumPy
# Many ML libraries (e.g., scikit-learn, NumPy) expect the target variable (y) to be a 2D array for operations like matrix multiplication.

# Normalization
X_mean = X.mean(axis=0)  # Column-wise mean
X_range = X.max(axis=0) - X.min(axis=0)  # Column-wise range
X_range[X_range == 0] = 1  # Avoid division by zero for constant features
X_scaled = (X - X_mean) / X_range  # Center and scale by range

# Add intercept term
# np.ones((X_scaled.shape[0], 1)) >>  Creates a column vector of one with the same number of rows as X_scaled.
# hstack >> horizontally stacks ones column & X_scaled.
X_scaled = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])
# print(X_scaled)

# Hyperparameters (optimized)
alpha = 0.01  # Learning rate
iterations = 2000  # Number of iterations
m = len(y)  # Number of samples

# Initialize theta
theta = np.zeros((X_scaled.shape[1], 1)) # X_scaled.shape[1] >> number of columns of X_scaled
# print(theta)

# Gradient Descent
cost_history = []
for i in range(iterations):
    predictions = X_scaled @ theta # @ is The matrix multiplication operator in NumPy.
    error = predictions - y
    gradient = (1/m) * X_scaled.T @ error
    theta -= alpha * gradient
    
    # Calculate cost (MSE)
    cost = (1/(2*m)) * np.sum(error**2) # np.sum(error**2): Sum of all squared errors.
    cost_history.append(cost) # Stores the current cost value in a list (cost_history) to tracks how the cost decreases over iterations.

# Final predictions
y_pred = X_scaled @ theta


print("Optimized Parameters:")
print(f"Learning rate (α): {alpha}")
print(f"Iterations: {iterations}")
print(f"\nFinal Coefficients (θ):\n{theta.ravel()}") # ravel(): Converts a 2D array (e.g., shape (n, 1)) to 1D (shape (n,)).


# Actual vs Predicted plot
plt.figure(figsize=(10, 6)) # Creates a plot figure (10x6 inches).
plt.scatter(y, y_pred, alpha=0.5, color='blue') # x-axis range, y-axis range, alpha: Opacity, color: Blue.
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2) # 'k--': Black (k) dashed (--) line, lw=2: Line width of 2, 
plt.xlabel("Actual Disease Progression")
plt.ylabel("Predicted Disease Progression")
plt.title("Full Dataset Performance")
plt.grid(True)
plt.savefig('results.png')
# plt.show()
