#AIM: DATA VISUALIZATION AND STORY TELLING
#Create meaningful visualization using data visualization tools
#Combine multiple visualizations to tell a compelling data story
#Present the findings and insights in a clear and concise manner
#Steps: Add a csv file with numerical column (eg. amount) and category column 
          

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import data
data = pd.read_csv('carsdata.csv')

# Exploratory Data Analysis
# Example: Plot distribution of a numerical variable
sns.histplot(data['price'], bins=20)
plt.title('Distribution of Numerical Column')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Visualization
# Example: Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='cars', y='price', data=data, hue='branch')
plt.title('Scatter Plot of Feature1 vs Feature2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(title='target') 
plt.show()


# Example: Side-by-side bar plots
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.barplot(x='colour', y='value1', data=data) #specify any numerical column
plt.title('Bar Plot 1')

plt.subplot(1, 2, 2)
sns.barplot(x='colour', y='value2', data=data) #specify any numerical column 
plt.title('Bar Plot 2')

plt.tight_layout()
plt.show()
