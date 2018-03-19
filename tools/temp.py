classes = ['fabric', 'foliage', 'glass', 'leather', 'metal', 'paper', 'plastic', 'stone', 'water', 'wood']
train_rate = 0.8
import sys
import os
import random

def which_class(file_path):
    for i in range(0, len(classes)):
        if classes[i] in file_path:
            return classes[i]
    return ""

def class2number(label):
    for i in range(0, len(classes)):
        if classes[i] == label:
            return i + 1
    return -1

def write_to_file(list_pair, out_file):
    for pair in list_pair:
        index = class2number(pair[1])
        with open(out_file, 'a') as f:
            f.write("{} {}\n".format(pair[0], str(index)))

data_folder = './DS_FMD/data/original'
data = []
for root, dirs, files in os.walk(data_folder):
    for filename in [f for f in files if f.endswith(".jpg")]:
        jpg = os.path.join(root, filename).replace("\\", "/")
        data.append([jpg, str(which_class(jpg))])

n_splits = 5
for i in range(1, n_splits + 1):
    list_train = []
    list_test = []
    for label in classes:
        pairs = [d for d in data if d[1] == label]
        random.shuffle(pairs)
        n_train = int(train_rate * len(pairs))
        train = pairs[0: n_train]
        test = pairs[n_train: len(pairs)]
        list_train.extend(train)
        list_test.extend(test)
    train_file = "./DS_FMD/splits/train_split_{}.txt".format(i)
    test_file = "./DS_FMD/splits/test_split_{}.txt".format(i)
    write_to_file(list_train, train_file)
    write_to_file(list_test, test_file)


