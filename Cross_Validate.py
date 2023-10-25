from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
diabetes = datasets.load_diabetes()
print(diabetes.keys())
print(diabetes.__sizeof__())
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()
cv_results = cross_validate(lasso, X, y, cv=3)
sorted(cv_results.keys())
print(cv_results['test_score'])

scores = cross_validate(lasso, X, y, cv=3,scoring=('r2', 'neg_mean_squared_error'),return_train_score=True)
print(scores['test_neg_mean_squared_error'])
print(scores['train_r2'])