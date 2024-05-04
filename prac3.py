#Apply feature-scaling techniques like standardization and 
# normalization to numerical features.

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Create a sample dataset
data = {
    'Feature1': [10, 20, 30, 40, 50],
    'Feature2': [0.1, 0.5, 1.5, 2.0, 2.5],
    'Feature3': [5, 10, 15, 20, 25]
}

df = pd.DataFrame(data)

# Split the dataset into features (X) and target (y)
X = df[['Feature1', 'Feature2', 'Feature3']]
y = df["Feature3"]  # Replace 'target_column' with the name of your target column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler_standard = StandardScaler()
X_train_standardized = scaler_standard.fit_transform(X_train)
X_test_standardized = scaler_standard.transform(X_test)

# Normalization (Min-Max scaling)
scaler_minmax = MinMaxScaler()
X_train_normalized = scaler_minmax.fit_transform(X_train)
X_test_normalized = scaler_minmax.transform(X_test)

print("Original features:")
print(X)
print("\nScaled features (standardized):")
print(X_train_standardized)
print("\nScaled features (normalized):")
print(X_train_normalized)	