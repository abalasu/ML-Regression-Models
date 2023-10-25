
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score

stu_df = pd.read_csv("D:\PythonData\student_percentage_1.csv")
stu_df_stat = stu_df.describe()

y = stu_df['Marks_Scored']
stu_features = ['Study_Hrs']
X = stu_df[stu_features]
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 1, train_size=0.75)
print(train_X)
print(test_X)
student_model = DecisionTreeRegressor(random_state=1,criterion='squared_error')
student_model.fit(train_X, train_y)
student_score_pred = student_model.predict(test_X)
plt.figure(figsize=(18,18))
tree.plot_tree(student_model, fontsize=9)
print('test_X')
print(test_X)
print('test_y')
print(test_y)
print('Pred_Score')
print(student_score_pred)
print("Mean Squared Error ", mean_squared_error(test_y, student_score_pred))
print("R2 Score ", r2_score(test_y, student_score_pred))

plt.show()