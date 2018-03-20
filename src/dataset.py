# Ex. X, y = dataset.get_data("keras_vgg16_fc2-original", "trainlist01")

import numpy
import pickle
import os
import glob

def get_data(dataset_name, feature_type, split_name):
    print("Get X. dataset_name: {}, feautre_type: {}, split_name: {}".format(dataset_name, feature_type, split_name))

    # if feature file existed, use it
    X_file = "./DS_{}/result/features/{}/{}.pkl".format(dataset_name, feature_type, split_name)
    y_file = "./DS_{}/result/labels/{}/{}.pkl".format(dataset_name, feature_type, split_name)
    if os.path.exists(X_file) and os.path.exists(y_file):
        with open(X_file, "rb") as fr:
            X = pickle.load(fr)
        with open(y_file, "rb") as fr:
            y = pickle.load(fr)
        return numpy.array(X), numpy.array(y)

    # else, mix feature from single feature files.
    split_file = "./DS_{}/splits/{}.txt".format(dataset_name, split_name)
    with open(split_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    X = []
    y = []
    for line in content:
        jpg_file = line.split()[0]
        label = line.split()[1]
        pkl_file = jpg_file.replace("data", "result/features").replace("jpg", "pkl")
        print("Read file {}".format(pkl_file))
        with open(pkl_file, "rb") as f:
            X.append(pickle.load(f))
            y.append(label)
    return numpy.array(X), numpy.array(y)

def confirm():
    X, y = get_data("FMD", "original", "train_split_1")
    print(numpy.shape(X))
    print(numpy.shape(y))

#confirm()