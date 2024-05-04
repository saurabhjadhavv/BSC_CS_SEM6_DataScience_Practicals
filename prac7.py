#AIM: Logistic Regression and Decision Tree
#Build a logistic regression model to predict a binary outcome.
#Evaluate the model’s performance using classification metrics (accuracy, precision, recall)
#Construct a decision tree model and interpret the decision rules for classification.


# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.datasets import load_iris
# You can replace this with your dataset

# Load the dataset (replace this with your dataset)
data = load_iris()
X = data.data
y = (data.target == 2).astype(int)  # Binary outcome, e.g., classifying if iris is of class 2 or not

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Logistic Regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

# Make predictions on the test set
logreg_predictions = logreg_model.predict(X_test)

# Evaluate Logistic Regression model performance
print("Logistic Regression Model Performance:")
print("Accuracy:", accuracy_score(y_test, logreg_predictions))
print("Precision:", precision_score(y_test, logreg_predictions))
print("Recall:", recall_score(y_test, logreg_predictions))
print("\nClassification Report:")
print(classification_report(y_test, logreg_predictions))

# Build Decision Tree model
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

# Make predictions on the test set using Decision Tree
tree_predictions = tree_model.predict(X_test)

# Evaluate Decision Tree model performance
print("\nDecision Tree Model Performance:")
print("Accuracy:", accuracy_score(y_test, tree_predictions))
print("Precision:", precision_score(y_test, tree_predictions))
print("Recall:", recall_score(y_test, tree_predictions))
print("\nClassification Report:")
print(classification_report(y_test, tree_predictions))
