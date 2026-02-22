# filename: reparametrization_validation.py
import numpy as np
import torch
import autograd
import h5py
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(1000, 1)
noise = 0.1 * np.random.randn(1000, 1)
y = 2 * X + 3 + noise

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, requires_grad=True)
y_tensor = torch.tensor(y, requires_grad=True)

# Define reparametrizations
X_affine = 2 * X_tensor + 1
y_affine = y_tensor

X_nonlinear = torch.tanh(X_tensor)
y_nonlinear = torch.tanh(y_tensor) + noise

X_symmetric = X_tensor ** 2
y_symmetric = y_tensor + noise

# Save to HDF5 file
with h5py.File('data/reparametrization_data.h5', 'w') as f:
    f.create_dataset('X_affine', data=X_affine.numpy())
    f.create_dataset('y_affine', data=y_affine.numpy())
    f.create_dataset('X_nonlinear', data=X_nonlinear.numpy())
    f.create_dataset('y_nonlinear', data=y_nonlinear.numpy())
    f.create_dataset('X_symmetric', data=X_symmetric.numpy())
    f.create_dataset('y_symmetric', data=y_symmetric.numpy())

    # Store shape information
    f.attrs['X_affine_shape'] = str(X_affine.shape)
    f.attrs['y_affine_shape'] = str(y_affine.shape)
    f.attrs['X_nonlinear_shape'] = str(X_nonlinear.shape)
    f.attrs['y_nonlinear_shape'] = str(y_nonlinear.shape)
    f.attrs['X_symmetric_shape'] = str(X_symmetric.shape)
    f.attrs['y_symmetric_shape'] = str(y_symmetric.shape)

# Output summary
print(f"X_affine shape: {X_affine.shape}")
print(f"y_affine shape: {y_affine.shape}")
print(f"X_nonlinear shape: {X_nonlinear.shape}")
print(f"y_nonlinear shape: {y_nonlinear.shape}")
print(f"X_symmetric shape: {X_symmetric.shape}")
print(f"y_symmetric shape: {y_symmetric.shape}")