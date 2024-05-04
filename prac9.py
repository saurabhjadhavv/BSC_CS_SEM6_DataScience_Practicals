#AIM: Principal Component Analysis (PCA)
#Perform PCA on dataset to reduce dimensionality.
#Evaluate the explained variance and select the appropriate number of principal components.
#Visualize the data in reduced-dimensional space.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate a simple 2D dataset (height and weight of individuals)
np.random.seed(42)
height = np.random.normal(160, 10, 100)  # mean=160, std=10
weight = height + np.random.normal(0, 5, 100)  # weight is correlated with height

# Create a 2D array with height and weight as features
data = np.column_stack((height, weight))

# Perform PCA
pca = PCA()
data_pca = pca.fit_transform(data)

# Visualize the original and reduced-dimensional data
plt.figure(figsize=(10, 5))

# Original Data
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c='blue', alpha=0.8)
plt.title('Original Data')
plt.xlabel('Height')
plt.ylabel('Weight')

# Reduced-Dimensional Data (using only the first principal component)
plt.subplot(1, 2, 2)
plt.scatter(data_pca[:, 0], np.zeros_like(data_pca[:, 0]), c='red', alpha=0.8)
plt.title('Reduced-Dimensional Data (1st Principal Component)')
plt.xlabel('Principal Component 1')

plt.tight_layout()
plt.show()