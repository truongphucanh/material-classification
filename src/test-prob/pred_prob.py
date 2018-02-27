import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC
clf = SVC(probability=True)
clf.fit(X, y)
print(clf.predict_proba([[-0.8, -1], [0.8, 1]]))
print(clf.predict([[-0.8, -1], [0.8, 1]]))
print(clf.classes_)
