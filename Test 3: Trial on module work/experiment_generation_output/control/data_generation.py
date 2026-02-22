# filename: data_generation.py
import numpy as np
import random
from tqdm import tqdm

# Generate synthetic data
features = []
targets = []

for _ in range(1000):
    feature1 = random.gauss(0, 1)
    feature2 = random.expovariate(0.5)
    features.append([feature1, feature2])
    targets.append(target)

features = np.array(features)
targets = np.array(targets)

# Example usage:
print(f"Features shape: {features.shape}")
print(f"Targets shape: {targets.shape}")