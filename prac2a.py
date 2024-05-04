#Handling outliers
#Using Z-score

import numpy as np
import matplotlib.pyplot as plt
#detecting outlier using zscore
datasets = [11, 17, 16, 15, 14, 13, 12, 256, 255, 250]
plt.hist(datasets)
plt.show()
outliers = []
threshold = 1  # 3rd standard deviation
mean = np.mean(datasets)
std = np.std(datasets)
for i in datasets:
    z_score = (i - mean) / std
    if z_score > threshold:
        outliers.append(i)
print("Outliers in dataset are:", outliers)