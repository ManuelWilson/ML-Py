from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

import numpy as np

import warnings
warnings.filterwarnings("ignore")


X, y = make_classification(n_samples=100, random_state=1)

# multiclass example (same code)
# X,y = load_iris().data, load_iris().target


rn = range(1,len(X))

kf = KFold(n_splits=5, shuffle=True) # 5 fold

parameter_space = {
    'hidden_layer_sizes': [(10,30,10),(20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

classification_report_list = []

for train_index, test_index in kf.split(rn):
    X_train, X_test, y_train, y_test = X[train_index,], X[test_index,] ,y[train_index], y[test_index]
    mlp_gs = MLPClassifier(max_iter=100)
    clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)  # cv is cross validation k fold 5 , n_jobs=-1 is for use all cpu's
    clf.fit(X_train, y_train) # X is train samples and y is the corresponding labels

    y_true, y_pred = y_test, clf.best_estimator_.predict(X_test)

    #capturar metricas pertinenetes
    #classification_report_list.append(classification_report(y_true, y_pred)) #as string
    classification_report_list.append(precision_recall_fscore_support(y_true, y_pred, average='micro'))


    '''
    # all optimization scores 
    print('Best parameters found:\n', clf.best_params_)
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    '''