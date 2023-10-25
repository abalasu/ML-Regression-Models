import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

# x = [1,2,3,5,6,7,8,9]
# y = [4,8,12,8,4,28,32,22]

# polyfit gives the co-efficients of the equation from the highest degree
z, residuals, rank, singular_values, rcond = np.polyfit(x, y, 3, full=True)
print(z)
# Residuals gives the variance between predicted and actuals that cannot be explained
print('residuals ', residuals)
# Rank - is the degree of the polynomial + 1
print('rank ', rank)
print('singular values ', singular_values)
print('rcond ', rcond)
# poly1d gives the actual regression equation
poly_model = np.poly1d(z)
print(poly_model)

#linspace returns a list that starts from the 1st parm, going till the 2nd parm. The 3rd parm gives the count of elements in the list
myline = np.linspace(1, 22, 100)
# Plot actual values
print(myline)
plt.scatter(x, y)
# Plot derived values from the model
plt.plot(myline, poly_model(myline))
# The R2 score tells how well the polynomial equation is in predicting the value of y. R2 score of +1 or -1 means a good fit. 0 means no relationship
print(r2_score(y, poly_model(x)))
plt.show()


