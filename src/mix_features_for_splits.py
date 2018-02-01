import pickle
import glob
import logging
import tools
import sys
import os
import numpy as np

LOW_INDEX = 0
HIGH_INDEX = 5

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

def mix_for_splits(feature_name):
    logger = logging.getLogger()
    logger.debug('Mix splits for feature {}...'.format(feature_name))
    feature_dir = '../bin/features/{}'.format(feature_name)
    if not os.path.exists(feature_dir):
        logger.error('!Error: folder {} not found.'.format(feature_dir))
        return -1
    for i in range(LOW_INDEX, HIGH_INDEX + 1):
        train_file = '../splits/trainlist0{}.txt'.format(i)
        train_pkl = '{}/trainlist0{}.pkl'.format(feature_dir, i)
        if os.path.exists(train_pkl):
            logger.debug('Feature file {} existed.'.format(train_pkl))
        else:
            logger.debug('Mixing feature for {}...'.format(train_file))
            X_train = get_mixed_features(train_file, feature_name)
            logger.debug('X_train shape: {}'.format(np.shape(X_train)))
            logger.debug('Saving feature to file {}...'.format(train_pkl))
            with open(train_pkl, 'wb') as f:
                pickle.dump(X_train, f)

        test_file = '../splits/testlist0{}.txt'.format(i)
        test_pkl = '{}/testlist0{}.pkl'.format(feature_dir, i)
        if os.path.exists(test_pkl):
            logger.debug('Feature file {} existed.'.format(test_pkl))
        else:
            logger.debug('Mixing feature for {}...'.format(test_file))
            X_test = get_mixed_features(test_file, feature_name)
            logger.debug('X_test shape: {}'.format(np.shape(X_test)))
            logger.debug('Saving feature to file {}...'.format(test_pkl))
            with open(test_pkl, 'wb') as f:
                pickle.dump(X_test, f)
    logger.debug('Done.')
    return 0

def main(argv):
    tools.config()
    if len(argv) < 2:
        print('Missing arguments: mix_features_for_splits.py `feature_name`. Ex. `mix_feature_for_splits.py keras_vgg16_fc2-edges`')
        return
    feature_name = argv[1]
    log_file = '../logs/mix_features_for_splits.log'
    logger = tools.get_logger(log_file, logging.INFO, logging.DEBUG)
    mix_for_splits(feature_name)
    return 0

if __name__ == '__main__':
    main(sys.argv)
