import pickle
import numpy as np

with open('trainlist00.pkl', 'rb') as f:
    feat = pickle.load(f)

print(np.shape(feat))
print(np.shape(feat[0]))
print(feat[0])