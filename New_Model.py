import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
from sklearn import tree

salary_df = pd.read_csv('d:/pythondata/processed.csv')
print(salary_df)
print(salary_df.describe())

X = salary_df[['Dept','Gender','Age','Location']]
y = salary_df['Salary']
train_X, test_X, train_y, test_y = train_test_split(X,y,train_size=0.8,random_state=0)
regr = linear_model.LinearRegression()
regr.fit(train_X,train_y)

print(f"intercept: {regr.intercept_}")
# The co-efficient here is the wt coef followed by the volume coef 
# The formula will be CO2 = wt coef * weight + vol coef * volume + intercept
print(f"slope: {regr.coef_}")
z = regr.predict(test_X)
print('The actuals are ')
print(test_y)
print('The prediction results are ')
print(z)

dt = DecisionTreeRegressor()
dt.fit(X,y)
z = dt.predict(test_X)
print('The actuals are ')
print(test_y)
print('The prediction results are ')
print(z)

plt.figure(figsize=(18,18))
tree.plot_tree(dt, fontsize=12)
plt.show()