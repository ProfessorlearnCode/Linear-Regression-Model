#imports for data handling
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#imports from sklearn library
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Features (X) and target (Y)
X = diabetes.data
Y = diabetes.target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, Y_train)

# Make predictions on the test data
Y_prediction = model.predict(X_test)

# Print model coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept: ", model.intercept_)

# Calculate and print the Mean Square Error and R² Score
mse = mean_squared_error(Y_test, Y_prediction)
r2 = r2_score(Y_test, Y_prediction)
print("Mean Square Error: %.2f" % mse)
print("R² Score: %.2f" % r2)

# Print actual vs predicted values
print("Actual Values:", Y_test)
print("Predicted Values:", Y_prediction)

# Scatter plot to visualize the relationship between actual and predicted values
sns.scatterplot(x=Y_test, y=Y_prediction, alpha=0.7)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# Residual plot to check for patterns
residuals = Y_test - Y_prediction
sns.scatterplot(x=Y_prediction, y=residuals, alpha=0.7)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(0, color='red', linestyle='--')
plt.show()
