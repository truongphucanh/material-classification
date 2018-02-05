import os
import pickle
import numpy

folder = "../bin/labels/"
for root, dirs, files in os.walk(folder):
    for name in [f for f in files if f.endswith(".pkl")]:
        full = os.path.join(root, name)
        print(full)
        with open(full, 'rb') as f:
            X = pickle.load(f)
        X = numpy.array(X)
        with open(full, 'wb') as f:
            pickle.dump(X, f)