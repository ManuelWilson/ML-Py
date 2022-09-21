from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

import numpy as np

import warnings
warnings.filterwarnings("ignore")

X,y = load_diabetes(return_X_y=True)
X = X[:, np.newaxis, 2] # only one feature

rn = range(1,len(X))

kf = KFold(n_splits=5, shuffle=True) # 5 fold

parameter_space = {
    'fit_intercept': [True, False]
}

regression_R_squared = []

for train_index, test_index in kf.split(rn):
    X_train, X_test, y_train, y_test = X[train_index,], X[test_index,] ,y[train_index], y[test_index]
    mlp_gs = linear_model.LinearRegression()
    clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)  # cv is cross validation k fold 5 , n_jobs=-1 is for use all cpu's
    clf.fit(X_train, y_train) # X is train samples and y is the corresponding labels

    y_true, y_pred = y_test, clf.best_estimator_.predict(X_test)
    regression_R_squared.append(r2_score(y_true, y_pred))

