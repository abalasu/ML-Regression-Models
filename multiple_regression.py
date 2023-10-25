import pandas as pd
from sklearn import linear_model

# Linear regression with two independent variables and 1 dependent variable
car_df = pd.read_csv('cardata.csv')
print(car_df)

X = car_df[["Weight","Volume"]]
y = car_df["CO2"]

regr = linear_model.LinearRegression()
regr.fit(X,y)
print(regr.score(X,y))

print(f"intercept: {regr.intercept_}")
# The co-efficient here is the wt coef followed by the volume coef 
# The formula will be CO2 = wt coef * weight + vol coef * volume + intercept
print(f"slope: {regr.coef_}")
cc = 1500
wt = 1250
z = regr.predict([[wt,cc]])
txt = 'The CO2 emission of a car with a cc of {0} and weight {1} kg is {2}'
print(txt.format(cc, wt, z))
