#Implement simple linear regression using a dataset.
#Explore and interpret the regression model coefficients and goodness-of-fit measures.
#Extend the analysis to multiple linear regression and assess the impact of additional predictors.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate example data for simple linear regression
np.random.seed(42)
X_simple = 2 * np.random.rand(100, 1)
y_simple = 4 + 3 * X_simple[:, 0] + np.random.randn(100)

# Split the data into training and testing sets for simple linear regression
X_simple_train, X_simple_test, y_simple_train, y_simple_test = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)

# Create a simple linear regression model
simple_model = LinearRegression()

# Fit the model to the training data for simple linear regression
simple_model.fit(X_simple_train, y_simple_train)

# Make predictions on the test data for simple linear regression
y_simple_pred = simple_model.predict(X_simple_test)

# Evaluate simple linear regression model
mse_simple = mean_squared_error(y_simple_test, y_simple_pred)
r2_simple = r2_score(y_simple_test, y_simple_pred)

# Print results for simple linear regression
print("Simple Linear Regression Results:")
print(f'Intercept: {simple_model.intercept_}')
print(f'Coefficient: {simple_model.coef_[0]}')
print(f'Mean Squared Error (MSE): {mse_simple}')
print(f'R-squared (R2): {r2_simple}')

# Visualize simple linear regression
plt.figure(figsize=(10, 5))
plt.scatter(X_simple_test, y_simple_test, color='black', label='Actual Data')
plt.plot(X_simple_test, y_simple_pred, color='blue', linewidth=3, label='Regression Line')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

# Generate example data for multiple linear regression
X_multi = 2 * np.random.rand(100, 2)
y_multi = 4 + 3 * X_multi[:, 0] + 2 * X_multi[:, 1] + np.random.randn(100)

# Split the data into training and testing sets for multiple linear regression
X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

# Create a multiple linear regression model
multi_model = LinearRegression()

# Fit the model to the training data for multiple linear regression
multi_model.fit(X_multi_train, y_multi_train)

# Make predictions on the test data for multiple linear regression
y_multi_pred = multi_model.predict(X_multi_test)

# Evaluate multiple linear regression model
mse_multi = mean_squared_error(y_multi_test, y_multi_pred)
r2_multi = r2_score(y_multi_test, y_multi_pred)

# Print results for multiple linear regression
print("\nMultiple Linear Regression Results:")
print(f'Intercept: {multi_model.intercept_}')
print(f'Coefficients: {multi_model.coef_}')
print(f'Mean Squared Error (MSE): {mse_multi}')
print(f'R-squared (R2): {r2_multi}')

# Visualize multiple linear regression
fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X_multi_test[:, 0], X_multi_test[:, 1], y_multi_test, c='black', marker='o')
ax1.set_xlabel('Independent Variable 1')
ax1.set_ylabel('Independent Variable 2')
ax1.set_zlabel('Dependent Variable (y)')
ax1.set_title('Actual Data')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_multi_test[:, 0], X_multi_test[:, 1], y_multi_pred, c='blue', marker='o')
ax2.set_xlabel('Independent Variable 1')
ax2.set_ylabel('Independent Variable 2')
ax2.set_zlabel('Predicted Variable (y)')
ax2.set_title('Multiple Linear Regression Prediction')