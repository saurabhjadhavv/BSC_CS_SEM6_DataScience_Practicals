import pandas as pd

# Create a sample dataset with a categorical variable
data = {
    'Feature1': ['A', 'B', 'A', 'C', 'B'],
    'Feature2': [10, 20, 30, 40, 50],
    'Target': [1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

# Display the original dataset
print("Original Dataset:")
print(df)

# Perform one-hot encoding using pandas get_dummies function
df_encoded = pd.get_dummies(df, columns=['Feature1'], prefix='Feature1')

# Display the dataset after one-hot encoding
print("\nDataset after One-Hot Encoding:")
print(df_encoded)