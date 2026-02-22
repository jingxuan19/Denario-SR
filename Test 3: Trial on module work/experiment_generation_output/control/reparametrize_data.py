# filename: reparametrize_data.py
import os
import pandas as pd
import numpy as np

# 1. Load data
features = pd.read_csv('data/raw/features.csv', index_col=0)
target = pd.read_csv('data/raw/target.csv', index_col=0)

# 2. Normalize features
features_norm = (features - features.mean()) / features.std()

# 3. Log-transform features
features_log = np.log(features + 1e-6)  # Adding small value to avoid log(0)

# 4. Combine and save reparametrized data
data_reparam = pd.concat([features_norm, features_log], axis=1)

# Create directory if it doesn't exist
os.makedirs('data/reparametrized', exist_ok=True)

# 5. Save to CSV
data_reparam.to_csv('data/reparametrized/reparametrized_data.csv', index=True)

# 6. Print summary statistics
print("\n📊 Summary Statistics of Reparametrized Data:")
print(data_reparam.describe())