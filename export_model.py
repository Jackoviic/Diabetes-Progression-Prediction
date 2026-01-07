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

# Normalization
X_mean = X.mean(axis=0)
X_range = X.max(axis=0) - X.min(axis=0)
X_range[X_range == 0] = 1
X_scaled = (X - X_mean) / X_range

# Add intercept term
X_scaled_biased = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

# Hyperparameters
alpha = 0.01
iterations = 2000
m = len(y)

# Initialize theta
theta = np.zeros((X_scaled_biased.shape[1], 1))

# Gradient Descent
for i in range(iterations):
    predictions = X_scaled_biased @ theta
    error = predictions - y
    gradient = (1/m) * X_scaled_biased.T @ error
    theta -= alpha * gradient

# Convert dataset to list of dicts for JSON
# keys will be columns of df
dataset_records = df.to_dict(orient='records')

# Prepare export data
model_params = {
    "theta": theta.flatten().tolist(), # Convert to 1D list
    "mean": X_mean.flatten().tolist(),
    "range": X_range.flatten().tolist(),
    "feature_names": df.drop('Y', axis=1).columns.tolist()
}

# Save Model Params
with open(f'{OUTPUT_DIR}/model_params.json', 'w') as f:
    json.dump(model_params, f, indent=2)

# Save Dataset
with open(f'{OUTPUT_DIR}/diabetes_data.json', 'w') as f:
    json.dump(dataset_records, f, indent=2)

print(f"Exported model_params.json and diabetes_data.json to {OUTPUT_DIR}/")
