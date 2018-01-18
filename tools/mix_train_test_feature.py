import numpy as np
import sys
import os
import pickle
import tools
import logging

def get_X(info_file, feature_name):
    logger = logging.getLogger()
    with open(info_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    X = []
    for line in content:
        folder = line.split()[0]
        curr_X = []
        feature_file = '../bin/features/{}/{}.pkl'.format(feature_name, folder)
        if not os.path.exists(feature_file):
            logger.debug('!Error: feature file {} not existed.'.format(feature_file))
            return []
        with open(feature_file, 'rb') as f:
            X.extend(pickle.load(f))
    return X

LOW_INDEX = 0 
HIGH_INDEX = 5
def mix(feature_name):
    logger = logging.getLogger()
    logger.debug('Mix train_test for feature {}...'.format(feature_name))
    feature_dir = '../bin/features/{}'.format(feature_name)
    if not os.path.exists(feature_dir):
        logger.error('!Error: folder {} not found.'.format(feature_dir))
        return -1
    for i in range(LOW_INDEX, HIGH_INDEX + 1):
        train_file = '../train_test/trainlist0{}.txt'.format(i)
        train_pkl = '{}/trainlist0{}.pkl'.format(feature_dir, i)
        if os.path.exists(train_pkl):
            logger.debug('Feature file {} existed.'.format(train_pkl))
        else:
            logger.debug('Mixing feature for {}...'.format(train_file))
            X_train = get_X(train_file, feature_name)
            logger.debug('X_train shape: {}'.format(np.shape(X_train)))
            logger.debug('Saving feature to file {}...'.format(train_pkl))
            with open(train_pkl, 'wb') as f:
                pickle.dump(X_train, f)

        test_file = '../train_test/testlist0{}.txt'.format(i)
        test_pkl = '{}/testlist0{}.pkl'.format(feature_dir, i)
        if os.path.exists(test_pkl):
            logger.debug('Feature file {} existed.'.format(test_pkl))
        else:       
            logger.debug('Mixing feature for {}...'.format(test_file))
            X_test = get_X(test_file, feature_name)
            logger.debug('X_test shape: {}'.format(np.shape(X_test)))
            logger.debug('Saving feature to file {}...'.format(test_pkl))
            with open(test_pkl, 'wb') as f:
                pickle.dump(X_test, f)
    logger.debug('Done.')
    return 0

def main(argv):
    logger = tools.get_logger('../logs/mix_train_test.log', file_level=logging.DEBUG, console_level=logging.DEBUG)
    logger.info('\n')
    logger.info('*'*50)
    if len(argv) <= 1:
        logger.debug('Not enough arguments.')
        return 0
    for i in range(1, len(argv)):
        feature_name = argv[i]
        mix(feature_name)

if __name__ == '__main__':
    main(sys.argv)