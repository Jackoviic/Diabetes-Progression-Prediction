import numpy as np
import pandas as pd
import json
import os

# Ensure web/public directory exists (will be created later if not, but good to handle data flow)
# We will save to a temporary location or the current dir first if web/ doesn't exist yet.
OUTPUT_DIR = 'web_data' 
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load dataset
df = pd.read_csv('assets/diabetes.csv')
X = df.drop('Y', axis=1).values
y = df[['Y']].to_numpy()

# Normalization (Standardization)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1.0 # Avoid division by zero
X_scaled = (X - X_mean) / X_std

# Add intercept term
X_scaled_biased = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

# Hyperparameters
alpha = 0.01
max_iters = 10000
epsilon = 1e-3
m = len(y)

# Initialize theta
theta = np.zeros((X_scaled_biased.shape[1], 1))

# Gradient Descent with convergence check
prev_cost = float('inf')

def compute_cost(X, y, theta, m):
    predictions = X @ theta
    error = predictions - y
    return (1/(2*m)) * np.sum(error ** 2)

for i in range(max_iters):
    predictions = X_scaled_biased @ theta
    error = predictions - y
    gradient = (1/m) * X_scaled_biased.T @ error
    theta -= alpha * gradient

    cost = compute_cost(X_scaled_biased, y, theta, m)
    
    if abs(prev_cost - cost) < epsilon:
        print(f"Converged at iteration {i+1}")
        break
    prev_cost = cost

# Define feature names (excluding Target Y and future Y_PRED)
feature_names = df.drop('Y', axis=1).columns.tolist()

# Calculate and add predictions to DataFrame
final_predictions = X_scaled_biased @ theta
df['Y_PRED'] = final_predictions.flatten()

# Convert dataset to list of dicts for JSON
# keys will be columns of df
dataset_records = df.to_dict(orient='records')

# Prepare export data
model_params = {
    "theta": theta.flatten().tolist(),
    "mean": X_mean.flatten().tolist(),
    "std": X_std.flatten().tolist(),
    "feature_names": feature_names
}

# Save Model Params
with open(f'{OUTPUT_DIR}/model_params.json', 'w') as f:
    json.dump(model_params, f, indent=2)

# Save Dataset
with open(f'{OUTPUT_DIR}/diabetes_data.json', 'w') as f:
    json.dump(dataset_records, f, indent=2)

print(f"Exported model_params.json and diabetes_data.json to {OUTPUT_DIR}/")
