#AIM: K-Means Clustering
#Apply the K-Means algorithm to group similar data points into clusters.
#Determine the optimal number of clusters using elbow method.
#Visualize the clustering results and analyze the cluster characteristics.

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  
# Adjust the number based on your system configuration

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
data, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Function to calculate the sum of squared distances for different values of k
def calculate_inertia(data, k_range):
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    return inertias

# Determine the optimal number of clusters using the elbow method
k_range = range(1, 11)
inertias = calculate_inertia(data, k_range)

# Plot the elbow curve
plt.plot(k_range, inertias, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Distances')
plt.show()

# Choose the optimal number of clusters based on the elbow point
optimal_k = 4  # You may adjust this based on the elbow point in the plot

# Apply KMeans clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(data)

# Visualize the clustering results
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolors='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('KMeans Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Analyze cluster characteristics
for i in range(optimal_k):
    cluster_points = data[labels == i]
    print(f"Cluster {i + 1} - Size: {len(cluster_points)}")
    print(f"Centroid: {kmeans.cluster_centers_[i]}")
    print(f"Sample points: {cluster_points[:5]}\n")
