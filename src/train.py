# Example: python train.py keras_vgg16_fc2-original
from sklearn.externals import joblib
import pickle
import sys
import time
import logging
import os
import numpy
import models
import dataset
import glob
from mkit import mlog

OVERWRITE = False

def train(feature_name, split_name):
    """Train for a training set from ./train_test folder with specific feature_name
    
    Arguments:
        feature_name {string} -- feature_name (Ex. 'keras_vgg16_fc2').
        split_name {string} -- split name (Ex. 'trainlist001').
    """
    # get logger
    logger = logging.getLogger()
    logger.info('Start training for '.format(feature_name, split_name))

    # get models
    clfs, lkernel, lC, ld, lg = models.get_models('../config/models_config.csv')

    # get data
    feature_dir = '../bin/features/{}'.format(feature_name)
    if not os.path.exists(feature_dir):
        logger.error('!Error: Feature folder {} not found. Please run feature extractor for this feature'.format(feature_dir))
        return
    X, y = dataset.get_data(feature_name, split_name)

    # fit data and save model to file
    models_dir = '../bin/models/{}/{}'.format(feature_name, split_name)
    fittime_file = '{}/time.csv'.format(models_dir)
    if not os.path.exists(models_dir):
        logger.info('Creating models folder {}...'.format(models_dir))
        os.makedirs(models_dir)
    if OVERWRITE:
        with open(fittime_file, 'wb') as fw:
            fw.write('Kernel,C,degree,gamma,fit time\n')
    for i, clf in enumerate(clfs):
        model_file = '{}_{}_{}_{}.pkl'.format(lkernel[i], lC[i], ld[i], lg[i])
        model_dir = '{}/{}'.format(models_dir, model_file)
        if (not OVERWRITE) and os.path.exists(model_dir):
            logger.info('Model {} existed.'.format(model_dir))
            continue
        logger.info('Fitting with model {}: {}'.format(i, model_file))
        start_time = time.time()
        clf.fit(X, y)
        fit_time = time.time() - start_time
        logger.info('Done. fit_time = {}'.format(fit_time))
        logger.info('Saving model {} to {} ...'.format(i, models_dir))
        joblib.dump(clf, model_dir)
        with open(fittime_file, 'a') as fw:
            fw.write('{},{},{},{},{}\n'.format(lkernel[i], lC[i], ld[i], lg[i], fit_time))

def train_for(feature_name, low, high):
    """Train for all train set in ./train_test folder with specific feature name
    
    Arguments:
        feature_name {string} -- feature name (Ex. 'keras_vgg16_fc2-original')
    """
    logger = logging.getLogger()
    feature_dir = '../bin/features/{}'.format(feature_name)
    if not os.path.exists(feature_dir):
        logger.error('!Error: Feature folder {} not found. Please run feature extractor for this feature'.format(feature_dir))
        return
    for i in range(low, high + 1):
        logger.info('>'*100)
        train(feature_name, 'trainlist0{}'.format(i))

def main(argv):
    mlog.config()
    logger = mlog.get_logger('../logs/train.log')
    if len(argv) <= 1:
        print('Missing arguments. Please try again.')
        print('Format: python train.py {feature name}')
        print('Example: python train.py keras_vgg16_fc2-original')
        return
    feature_name = argv[1]
    LOW_INDEX = 1
    HIGH_INDEX = 5
    if len(argv) >= 4:
        LOW_INDEX = int(argv[2])
        HIGH_INDEX = int(argv[3])
    train_for(feature_name, LOW_INDEX, HIGH_INDEX)

if __name__ == '__main__':
    main(sys.argv)
