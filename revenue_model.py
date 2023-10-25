import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Linear regression with two independent variables and 1 dependent variable
rev_df = pd.read_excel('d:/pythondata/cgi_rev/Step1.xlsx')
print(rev_df)

rev_features = ['B-Standard','B-Shift Hrs','B-OnCall','N-Standard','N-Shift Hrs','N-OnCall','I-Standard','I-Shift Hrs','I-OnCall','Working_day']

rev_df1 = rev_df.groupby('WK')[['B-Standard','B-Shift Hrs','B-OnCall','N-Standard','N-Shift Hrs','N-OnCall','I-Standard','I-Shift Hrs','I-OnCall','Working_day','Total Revenue']].sum()
rev_X = rev_df1[rev_features]
rev_y = rev_df1['Total Revenue']
rev_df1.to_excel('d:/pythondata/cgi_rev/grouped.xlsx')
train_X, test_X, train_y, test_y = train_test_split(rev_X,rev_y,train_size=0.8,random_state=1)
add_model = LinearRegression()
add_model.fit(train_X,train_y)
print(add_model.score(train_X,train_y))

print(f"intercept: {add_model.intercept_}")
# The co-efficient here is the wt coef followed by the volume coef 
# The formula will be CO2 = wt coef * weight + vol coef * volume + intercept
print(f"slope: {add_model.coef_}")
z = add_model.predict(test_X)
print('R2 Score ', r2_score(test_y,z))
print('Mean Absolute Error ', mean_absolute_error(test_y, z))
print('Mean Squared Error ', mean_squared_error(test_y,z))  