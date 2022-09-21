print('inicio')

from sklearn import svm
import pickle


X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
clf.predict([[2., 2.]])


#SAVE MODEL
pickle.dump(clf, open('../svm-example.sav', 'wb'))

#LOAD MODEL
loaded_model = pickle.load(open('../svm-example.sav', 'rb'))
result = loaded_model.predict([[2., 2.]])
print(result)