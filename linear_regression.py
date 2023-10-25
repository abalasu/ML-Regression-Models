import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 1. Linear regression with scipy
x = [1,2,3,4]
y = [3,7,6,9]

slope, intercept, r, p, std_err = stats.linregress(x, y)

print('slope ', slope)
print('intercept ', intercept)
print('r ', r)
print('R2', r**2)
print('p ', p)
print('std error ', std_err)

def myfunc(x):
    return slope*x + intercept

# map method converts a value based on a function or dictionary
mymodel = list(map(myfunc, x))
# Plotting actual values
plt.scatter(x,y)
# Plotting derived values
plt.plot(x, mymodel)
plt.show()

