import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Excel file
data = pd.read_excel("/content/cleaned_student_performance.xlsx")

# Show actual column names
print("Columns available:", data.columns)

# Use correct column names (adjust them based on the print output)
X = data['study_hours_per_day'].values.astype(float)            # Example column name
y = data['exam_score'].values.astype(float)      # Example column name

# Linear Regression from scratch
mean_x = np.mean(X)
mean_y = np.mean(y)
m = np.sum((X - mean_x) * (y - mean_y)) / np.sum((X - mean_x)**2)
b = mean_y - m * mean_x

# Prediction
y_pred = m * X + b

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('Study Hours per Day')
plt.ylabel('Exam Score')
plt.title('Linear Regression - Student Performance')
plt.legend()
plt.grid(True)
plt.show()

### for ploting all the graph 
import pandas as pd

df = pd.read_excel("/cleaned_student_performance.xlsx")
display(df.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Selecting the independent (X) and dependent (y) variables
X = df[['study_hours_per_day']]
y = df['exam_score']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Printing the model's coefficients
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

import matplotlib.pyplot as plt

# Making predictions on the test set
y_pred = model.predict(X_test)

# Plotting the results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title('Study Hours vs. Exam Score')
plt.xlabel('Study Hours per Day')
plt.ylabel('Exam Score')
plt.legend()
plt.show()
