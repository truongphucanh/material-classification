""""Training svm models"""

import pickle
import sys
import time
import logging
import os
import csv
from sklearn import svm
from sklearn.externals import joblib
import numpy as np
import tools
import glob
import copy

OVERWRITE_MODEL = False
MODELS_CONFIG_FILE = '../config/models_config.csv'
FEATURES_DIR_FORMAT = '../bin/features/{}' #.format(feature_name)
MODELS_DIR_FORMAT = '../bin/models/{}/trainlist0{}' #.format(feature_name, trainset_index)
X_FILE_FORMAT = '../bin/features/{}/trainlist0{}.pkl' #.format(feature_name, trainset_index)
Y_FILE_FORMAT = '../bin/labels/trainlist0{}.pkl' #.format(trainset_index)
FIT_TIME_ROW_FORMAT = '{},{},{},{},{}\n'#.format(kernel,C,d,g,time)
LOG_FILE = '../logs/train.log'
LOW_TRAINSET_INDEX = 0
HIGH_TRAINSET_INDEX = 6

def get_mixed_features(split_file, feature_name):
    logger = logging.getLogger()
    with open(split_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    X = []
    for line in content:
        folder = line.split()[0]
        feature_folder = '../bin/features/{}/{}'.format(feature_name, folder)
        logger.debug('Run on feature folder: {}'.format(feature_folder))
        for pkl_file in glob.glob('{}/*.pkl'.format(feature_folder)):
            with open(pkl_file, 'rb') as f:
                X.append(pickle.load(f))
    return X

def get_models(models_config_file):
    """Get list of model with different parameters for training.

    Parameters
    ----------
    models_config_file: string
        models config file name (must be *.csv) with format for each row `kernel, C, degree, gamma`

    Returns
    -------
    models: list
        List of sklearn.svm models with differenct parameters. 
    """
    logger = logging.getLogger()
    models = []
    list_kernel = []
    list_C = []
    list_degree = []
    list_gamma = []
    if not models_config_file.endswith('.csv'):
        logger.error('!Error: models config file must be *.csv, but it is {}'.format(models_config_file))
        return models, list_kernel, list_C, list_degree, list_gamma
    if not os.path.exists(models_config_file):
        logger.error('!Error: File not found {}'.format(models_config_file))
        return models, list_kernel, list_C, list_degree, list_gamma
    with open(models_config_file) as f:
        reader = csv.reader(f)
        rows = list(reader)
        for row in rows:
            if len(row) < 4:
                logger.error('!Error: Invalid format in models config file {}. Please folow format \"kernel, C, degree, gamma\"'.format(models_config_file))
                return []
            kernel = row[0]
            C = row[1]
            degree = row[2]
            gamma = row[3]
            model = svm.LinearSVC()
            if kernel == 'default':
                model = svm.SVC(probability=True)
            elif kernel == 'linear':
                model = svm.SVC(kernel=kernel, C=float(C), probability=True)
            elif kernel == 'rbf':
                model = svm.SVC(kernel=kernel, C=float(C), gamma=float(gamma), probability=True)
            elif kernel == 'poly':
                model = svm.SVC(kernel=kernel, C=float(C), degree=float(degree), probability=True)
            else:
                logger.error('!Error: Invalid kernel. Kernel must be one of `linear, rbf or poly` but it is {}').format(kernel)
            models.append(model)
            list_kernel.append(kernel)
            list_C.append(C)
            list_degree.append(degree)
            list_gamma.append(gamma)
    return models, list_kernel, list_C, list_degree, list_gamma

def get_train_data(feature_name, trainset_index):
    """Get training data
    
    Arguments:
        feature_name {string} -- feature name (Ex. 'keras_vgg16_fc2')
        trainset_index {int} -- index of training set in folder `./train_test/`
    
    Returns:
        X, y -- List of features (X) and labels (y)
    """
    logger = logging.getLogger()
    logger.info('Getting training data for training set {} with feature {}...'.format(trainset_index, feature_name))
    y_file = Y_FILE_FORMAT.format(trainset_index)
    X = []
    y = []
    if not os.path.exists(y_file):
        logger.error('!Error: Label file {} not found.'.format(y_file))
        return X, y
    logger.info('Mix features...')
    X = get_mixed_features('../splits/trainlist0{}.txt'.format(trainset_index), feature_name)

    logger.info('X shape: {}'.format(np.shape(X)))
    logger.info('Getting y from {}...'.format(y_file))
    with open(y_file, 'rb') as fr:
        y = pickle.load(fr)
    logger.info('y shape: {}'.format(np.shape(y)))
    return X, y

def train(feature_name, trainset_index):
    """Train for a training set from ./train_test folder with specific feature_name
    
    Arguments:
        feature_name {string} -- feature_name (Ex. 'keras_vgg16_fc2').
        trainset_index {int} -- index of training file in ./train_set folder (Ex. 0).
    """
    logger = logging.getLogger()
    logger.info('Start training for train set {} with feature {}'.format(trainset_index, feature_name))
    models, lkernel, lC, ld, lg = get_models(MODELS_CONFIG_FILE)
    feature_dir = FEATURES_DIR_FORMAT.format(feature_name)
    if not os.path.exists(feature_dir):
        logger.error('!Error: Feature folder {} not found. Please run feature extractor for this feature'.format(feature_dir))
        return
    X, y = get_train_data(feature_name, trainset_index)
    models_dir = MODELS_DIR_FORMAT.format(feature_name, trainset_index)
    fittime_file = '{}/time.csv'.format(models_dir)
    if not os.path.exists(models_dir):
        logger.info('Creating models folder {}...'.format(models_dir))
        os.makedirs(models_dir)
    if OVERWRITE_MODEL:
        with open(fittime_file, 'wb') as fw:
            fw.write('Kernel,C,degree,gamma,fit time\n')
    for i, model in enumerate(models):
        model_file = '{}_{}_{}_{}.pkl'.format(lkernel[i], lC[i], ld[i], lg[i])
        model_dir = '{}/{}'.format(models_dir, model_file)
        if (not OVERWRITE_MODEL) and os.path.exists(model_dir):
            logger.info('Model {} existed.'.format(model_dir))
            continue
        logger.info('Fitting with model {}: {}'.format(i, model_file))
        start_time = time.time()
        clone = copy.deepcopy(model)
        clone.fit(X, y)
        fit_time = time.time() - start_time
        logger.info('Done. fit_time = {}'.format(fit_time))
        logger.info('Saving model {} to {} ...'.format(i, models_dir))
        joblib.dump(clone, model_dir)
        with open(fittime_file, 'a') as fw:
            fw.write(FIT_TIME_ROW_FORMAT.format(lkernel[i], lC[i], ld[i], lg[i], fit_time))

def train_for(feature_name):
    """Train for all train set in ./train_test folder with specific feature name
    
    Arguments:
        feature_name {string} -- feature name (Ex. 'keras_vgg16_fc2')
    """
    logger = logging.getLogger()
    logger.info('Start runing for feature {}...'.format(feature_name))
    feature_dir = FEATURES_DIR_FORMAT.format(feature_name)
    if not os.path.exists(feature_dir):
        logger.error('!Error: Feature folder {} not found. Please run feature extractor for this feature'.format(feature_dir))
        return
    for i in range(LOW_TRAINSET_INDEX, HIGH_TRAINSET_INDEX):
        logger.info('>'*100)
        train(feature_name, i)
        logger.info('>'*100)

def main(argv):
    tools.config()
    if OVERWRITE_MODEL:
        with open(LOG_FILE, 'wb') as fw:
            fw.write('')
    logger = tools.get_logger(LOG_FILE, logging.INFO, logging.DEBUG)
    if len(argv) == 1:
        logger.debug('Try again with feature name args. Ex. `python train.py keras_vgg16_fc2 alexnet`')
        return
    for i in range(1, len(argv)):
        logger.info('*'*100)
        feature_name = argv[i]
        train_for(feature_name)
        logger.info('*'*100)

if __name__ == '__main__':
    main(sys.argv)
