import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC
clf = SVC(probability=True)
clf.fit(X, y)
print(clf.predict([[-0.8, -1]]))
clf.probability=True
result = (clf.predict_proba([[-0.8, 1]]))[0]
prob_per_class_dictionary = dict(zip(clf.classes_, result))
print(prob_per_class_dictionary)
print(prob_per_class_dictionary[1])