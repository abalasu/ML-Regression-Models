import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

num_samples = 1000
X_train = np.random.rand(num_samples, 2)
print(X_train)
y_train = (X_train[:, 0] + X_train[:, 1]) * 2

add_model = LinearRegression()
add_model.fit(X_train,y_train)
print(f"intercept: {add_model.intercept_}")
print(f"slope: {add_model.coef_}")
test_X = np.array([[1, 2], [0.3, 0.4],[323,541]])
test_y = np.array([3,0.7,864])
z = add_model.predict(test_X)
print(z)
print('R2 Score ', r2_score(test_y,z))
print('Mean Absolute Error ', mean_absolute_error(test_y, z))
print('Mean Squared Error ', mean_squared_error(test_y,z))  

"""
num_samples = 1000
X_train = np.random.rand(num_samples, 2)
y_train = X_train[:, 0] - X_train[:, 1]

add_model = LinearRegression()
add_model.fit(X_train,y_train)

test_X = np.array([[1, 2], [0.3, 0.4],[323,541]])
test_y = np.array([-1,-0.1,-218])
z = add_model.predict(test_X)
print(z)
print('R2 Score ', r2_score(test_y,z))
print('Mean Absolute Error ', mean_absolute_error(test_y, z))
print('Mean Squared Error ', mean_squared_error(test_y,z))

test_X = pd.DataFrame([2,3,4,5,6])
test_y = pd.DataFrame([4,6,6,7,9])

add_model = LinearRegression()
add_model.fit(test_X,test_y)
print(f"intercept: {add_model.intercept_}")
print(f"slope: {add_model.coef_}")
z = add_model.predict(test_X)
print('R2 Score ', r2_score(test_y,z))
print('Mean Absolute Error ', mean_absolute_error(test_y, z))
print('Mean Squared Error ', mean_squared_error(test_y,z))

"""