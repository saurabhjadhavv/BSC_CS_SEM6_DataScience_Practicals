#Using IQR

import numpy as np
import matplotlib.pyplot as plt
#detecting outlier using IQR

#SORT THE DATA
#Calculate q1(25%) and q3(75%)
#Iqr(q3-q1)
#Lower fence(q1-1.5(iqr))
#Upper fence(q3+1.5(iqr))

datasets = [11, 17, 16, 15, 14, 13, 12, 256, 255, 250]
dataset=sorted(datasets)
print(dataset)
q1,q3=np.percentile(dataset,[25,75])
print(q1,q3)
iqr=q3-q1
print(iqr)
lower_fence=q1-(1.5*iqr)
upper_fence=q3+(1.5*iqr)
print(lower_fence,upper_fence)

import seaborn as sns
sns.boxplot(datasets)

plt.show()