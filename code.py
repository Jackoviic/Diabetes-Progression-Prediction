import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# 1) Load dataset
# =========================
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "assets", "diabetes.csv")

df = pd.read_csv(csv_path)

X = df.drop('Y', axis=1).values
y = df[['Y']].to_numpy()
m = len(y)

# =========================
# 2) Train / Test split
# =========================
test_ratio = 0.2
indices = np.random.permutation(m)

test_size = int(m * test_ratio)
test_idx  = indices[:test_size]
train_idx = indices[test_size:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# =========================
# 3) Feature scaling (Standardization)
# =========================
X_mean = X_train.mean(axis=0)
X_std  = X_train.std(axis=0)
X_std[X_std == 0] = 1.0

X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled  = (X_test  - X_mean) / X_std

X_train_scaled = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])
X_test_scaled  = np.hstack([np.ones((X_test_scaled.shape[0], 1)),  X_test_scaled])

# =========================
# 4) Hyperparameters + convergence
# =========================
alpha = 0.01
max_iters = 3000
epsilon = 1e-3

theta = np.zeros((X_train_scaled.shape[1], 1))

# =========================
# 5) Cost function
# =========================
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    error = predictions - y
    cost = (1/(2*m)) * np.sum(error ** 2)
    return cost

# =========================
# 6) Gradient Descent with automatic convergence (train only)
# =========================
cost_history = []
prev_cost = float('inf')

for i in range(max_iters):
    predictions = X_train_scaled @ theta
    error = predictions - y_train
    gradient = (1/len(y_train)) * (X_train_scaled.T @ error)
    theta -= alpha * gradient

    cost = compute_cost(X_train_scaled, y_train, theta)
    cost_history.append(cost)

    # convergence check
    if abs(prev_cost - cost) < epsilon:
        print(f"Converged at iteration {i+1} with cost {cost:.4f}")
        break

    prev_cost = cost
else:
    print(f"Reached max iterations {max_iters} with cost {cost_history[-1]:.4f}")

# =========================
# 7) Final predictions (train & test)
# =========================
y_train_pred = X_train_scaled @ theta
y_test_pred  = X_test_scaled  @ theta

train_mse = compute_cost(X_train_scaled, y_train, theta) * 2
test_mse  = compute_cost(X_test_scaled,  y_test,  theta) * 2

print("Optimized Parameters:")
print(f"Learning rate (α): {alpha}")
print(f"Used iterations: {len(cost_history)}")
print(f"Train MSE: {train_mse:.4f}")
print(f"Test  MSE: {test_mse:.4f}")
print(f"\nFinal Coefficients (θ):\n{theta.ravel()}")

# =========================
# 8) Other plots
# =========================
os.makedirs('results', exist_ok=True)

# (a) Test set performance
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6, color='blue', label='Test Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'k--', lw=2, label='Perfect Prediction')
plt.xlabel("Actual Disease Progression (Test)")
plt.ylabel("Predicted Disease Progression (Test)")
plt.title("Test Set Performance (Standardized + Convergence GD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/test_performance.png')
# plt.show()

# (b) Train set performance
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, alpha=0.6, color='green', label='Train Predictions')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
         'k--', lw=2, label='Perfect Prediction')
plt.xlabel("Actual Disease Progression (Train)")
plt.ylabel("Predicted Disease Progression (Train)")
plt.title("Train Set Performance (Standardized + Convergence GD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/train_performance.png')
# plt.show()

# (c) Cost history plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cost_history) + 1), cost_history, 'b-')
plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.title("Cost Function Convergence (Train)")
plt.grid(True)
plt.tight_layout()
plt.savefig('results/cost_history.png')
# plt.show()
