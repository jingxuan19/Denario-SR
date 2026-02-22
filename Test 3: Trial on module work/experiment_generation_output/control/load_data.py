# filename: load_data.py
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data/raw_data.csv")

# Split into features and target
X = data[[