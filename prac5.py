#Perform one-way ANOVA to compare across multiple groups.
#Conduct post-hoc tests to identify significant differences between group means.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Create a sample dataset with multiple groups
data = pd.DataFrame({
    'Group_A': np.random.normal(0, 1, 100),
    'Group_B': np.random.normal(1, 1, 100),
    'Group_C': np.random.normal(2, 1, 100),
})

# Perform one-way ANOVA
f_statistic, p_value = f_oneway(data['Group_A'], data['Group_B'], data['Group_C'])

print(f'One-way ANOVA results: F-statistic = {f_statistic}, p-value = {p_value}')

# Box plot to compare group distributions
data.boxplot(grid=False)
plt.title('Box plot of Group Distributions')
plt.ylabel('Values')
plt.show()

# Conduct post-hoc tests (Tukey's HSD)
posthoc = pairwise_tukeyhsd(data.values.flatten(), np.repeat(data.columns, len(data)))
print(posthoc)

# Tukey's HSD plot
posthoc.plot_simultaneous()
plt.title('Tukey\'s HSD - Significant Differences')
plt.show()