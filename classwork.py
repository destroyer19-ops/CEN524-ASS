import numpy as np
from sklearn.linear_model import LinearRegression

# Define the function to calculate v1 and v2
def calculate_v1_v2(vt, v4, r1, r2):
    v1 = (v4 - r1) / (r1 + r2)
    v2 = (vt * r2) / (r1 + r2)
    return v1, v2

# Define some sample data
vt = 10
v4 = 20
r1 = np.array([1, 2, 3, 4, 5])
r2 = np.array([2, 3, 4, 5, 6])

# Calculate v1 and v2
v1, v2 = calculate_v1_v2(vt, v4, r1, r2)

# Reshape r1 to match the input format expected by sklearn
r1 = r1.reshape(-1, 1)

# Train a linear regression model to relate r1 and r2 to v1
regression_model = LinearRegression()
regression_model.fit(r1, v1)

# Predict v1 using the trained model
v1_predicted = regression_model.predict(r1)

# Print the coefficients and intercept of the linear regression model
print('Coefficient:', regression_model.coef_[0])
print('Intercept:', regression_model.intercept_)

# Print the predicted values of v1
print('Predicted v1:', v1_predicted)
