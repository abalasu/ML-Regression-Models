from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

stu_df = pd.read_csv("D:\PythonData\student_percentage_1.csv")
stu_df_stat = stu_df.describe()

y = stu_df['Marks_Scored']
stu_features = ['Study_Hrs']
X = stu_df[stu_features]
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 1, train_size=0.75)
print(train_X)
student_model = KNeighborsRegressor(n_neighbors = 2)
student_model.fit(train_X, train_y)
student_score_pred = student_model.predict(test_X)
print('test_X')
print(test_X)
print('test_y')
print(test_y)
print('Pred_Score')
print(student_score_pred)
print("Mean Squared Error ", mean_squared_error(test_y, student_score_pred))
print("R2 Score ", r2_score(test_y, student_score_pred))
