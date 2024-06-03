import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define your data (x and y)
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Example input features
y = np.array([2, 4, 5, 4, 5])  # Example target variable

# Initialize and train the linear regression model
regression_model = LinearRegression()
regression_model.fit(x, y)

# Predict
y_predicted = regression_model.predict(x)

# Model evaluation
mse = mean_squared_error(y, y_predicted)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_predicted)

# Print the evaluation metrics
print('Slope:', regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('MSE:', mse)
print('Root mean squared error:', rmse)
print('R2 score:', r2)
