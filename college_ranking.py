import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 

college_train_df = pd.read_excel('d:/pythondata/college_ranking_train.xlsx')
print(college_train_df)
print(college_train_df.describe())

X = college_train_df[['Age of college (A)','Number of courses offered (B)','Number of professors with PhD (C)','Number of Students (D)','Number of research papers presented (E)']]
y = college_train_df['College Ranking (Actual)']
train_X, test_X, train_y, test_y = train_test_split(X,y,train_size=0.8,random_state=0)
regr = linear_model.LinearRegression()
regr.fit(X,y)

print(f"intercept: {regr.intercept_}")
# The co-efficient here is the wt coef followed by the volume coef 
# The formula will be CO2 = wt coef * weight + vol coef * volume + intercept
print(f"slope: {regr.coef_}")
college_test_df = pd.read_excel('d:/pythondata/college_ranking_test.xlsx')
X = college_train_df[['Age of college (A)','Number of courses offered (B)','Number of professors with PhD (C)','Number of Students (D)','Number of research papers presented (E)']]
y = college_train_df['College Ranking (Actual)']

z = regr.predict(test_X)
print('The actuals are ')
print(test_y)
print('The prediction results are ')
print(z)

print('R2 Score ', r2_score(test_y,z))
print('Mean Absolute Error ', mean_absolute_error(test_y, z))
print('Mean Squared Error ', mean_squared_error(test_y,z))