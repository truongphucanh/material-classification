# Ex. X, y = dataset.get_data('keras_vgg16_fc2-original', 'trainlist01')

import numpy
import pickle
import os
import glob

def get_X(feature_name, split_name):
    print("Getting X for {} {}.".format(feature_name,  split_name))

    # if feature file existed, use it
    X_file = '../bin/features/{}/{}.pkl'.format(feature_name, split_name)
    if os.path.exists(X_file):
        with open(X_file, 'rb') as fr:
            X = pickle.load(fr)
        return numpy.array(X)

    # else, mix feature from single feature files.
    with open('../splits/{}.txt'.format(split_name)) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    X = []
    for line in content:
        folder = line.split()[0]
        feature_folder = '../bin/features/{}/{}'.format(feature_name, folder)
        print('Mix X in folder: {}'.format(feature_folder))
        for pkl_file in glob.glob('{}/*.pkl'.format(feature_folder)):
            with open(pkl_file, 'rb') as f:
                X.append(pickle.load(f))
    return numpy.array(X)

def get_y(split_name):
    y_file = '../bin/labels/{}.pkl'.format(split_name)
    if not os.path.exists(y_file):
        print('!Error: Label file {} not found.'.format(y_file))
        return
    print('Getting y from {}...'.format(y_file))
    with open(y_file, 'rb') as fr:
        y = pickle.load(fr)
    return numpy.array(y)

def get_data(feature_name, split_name):
    print('Getting training data for training set {} with feature {}...'.format(split_name, feature_name))
    X_file = '../bin/features/{}/{}.pkl'.format(feature_name, split_name)
    y_file = '../bin/labels/{}.pkl'.format(split_name)
    
    # get X
    X = get_X(feature_name, split_name)
    print('X shape: {}'.format(numpy.shape(X)))

    # get y
    y = get_y(split_name)
    print('y shape: {}'.format(numpy.shape(y)))

    return X, y

def confirm():
    X, y = get_data('keras_vgg16_fc2-original', 'trainlist02')
    print(numpy.shape(X))
    print(numpy.shape(y))

# confirm()