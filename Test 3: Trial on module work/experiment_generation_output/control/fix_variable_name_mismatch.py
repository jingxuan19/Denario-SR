# filename: fix_variable_name_mismatch.py
import pandas as pd
import random

features = []
targets = []
for _ in range(1000):
    feature1 = random.gauss(0, 1)
    feature2 = random.expovariate(0.5)
    features.append([feature1, feature2])
    target = 2 * feature1 + 3 * feature2 + random.gauss(0, 0.1)
    targets.append(target)

data = pd.DataFrame(features, columns=["feature1", "feature2"])
data["target"] = targets  # Correct usage