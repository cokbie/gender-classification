# from sklearn import classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# data and labels
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Classifiers
clf_1 = DecisionTreeClassifier()
clf_2 = GaussianNB()
clf_3 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

# train classifiers on data
clf_1 = clf_1.fit(X, Y)
clf_2 = clf_2.fit(X, Y)
clf_3 = clf_3.fit(X, Y)

# test with a set of new data
prediction_1 = clf_1.predict([[190, 70, 43]])
prediction_2 = clf_2.predict([[190, 70, 43]])
prediction_3 = clf_3.predict([[190, 70, 43]])

# print results
print('Gender prediction results for [190, 70, 43]')
print('DecisionTree: ', prediction_1)
print('GaussianNB: ', prediction_2)
print('LogisticRegression: ', prediction_3)

print('Model testing with same data')

# test with same data
prediction_1 = clf_1.predict(X)
score_1 = accuracy_score(Y, prediction_1) * 100
print('Accuracy score for DecisionTree is {}'.format(score_1))

prediction_2 = clf_2.predict(X)
score_2 = accuracy_score(Y, prediction_2) * 100
print('Accuracy score for GaussianNB is {}'.format(score_2))

prediction_3 = clf_3.predict(X)
score_3 = accuracy_score(Y, prediction_3) * 100
print('Accuracy score for LogisticRegression is {}'.format(score_3))

# print the best classifier
best_index = np.argmax([score_1, score_2, score_3])
clf = {0:'DecisionTree', 1:'GaussianNB', 2:'LogisticRegression'}
print('Best gender classifier is {}'.format(clf[best_index]))
