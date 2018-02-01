"""Testing process"""
import glob
import pickle
import sys
import time
import logging
import os
import csv
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import tools

OVERWRITE_MODE = False
LOW_TESTSET_INDEX = 0
HIGH_TESTSET_INDEX = 1
FEATURES_DIR_FORMAT = '../bin/features/{}' #.format(feature_name)
MODELS_DIR_FORMAT = '../bin/models/{}/trainlist0{}' #.format(feature_name, trainset_index)
TEST_RESULT_FOLDER_FORMAT = '../bin/test/{}/testlist0{}/{}' #.format(feature_name, testset_index, model_name)
ACCURAY_FILE_FORMAT = '../bin/test/{}/testlist0{}'
X_FILE_FORMAT = '../bin/features/{}/testlist0{}.pkl' #.format(feature_name, testset_index)
Y_FILE_FORMAT = '../bin/labels/testlist0{}.pkl' #.format(testset_index)
LOG_FILE = '../logs/test.log'

def get_test_data(feature_name, testset_index):
    """Get training data
    
    Arguments:
        feature_name {string} -- feature name (Ex. 'keras_vgg16_fc2')
        testset_index {int} -- index of training set in folder `./train_test/`
    
    Returns:
        X, y -- List of features (X) and labels (y)
    """
    logger = logging.getLogger()
    logger.info('Getting training data for training set {} with feature {}...'.format(testset_index, feature_name))
    X_file = X_FILE_FORMAT.format(feature_name, testset_index)
    y_file = Y_FILE_FORMAT.format(testset_index)
    X = []
    y = []
    if not os.path.exists(X_file):
        logger.error('!Error: Feature file {} not found.'.format(X_file))
        return X, y
    if not os.path.exists(y_file):
        logger.error('!Error: Label file {} not found.'.format(y_file))
        return X, y
    logger.info('Getting X from {}...'.format(X_file))
    with open(X_file, 'rb') as fr:
        X = pickle.load(fr)
    logger.info('X shape: {}'.format(np.shape(X)))
    logger.info('Getting y from {}...'.format(y_file))
    with open(y_file, 'rb') as fr:
        y = pickle.load(fr)
    logger.info('y shape: {}'.format(np.shape(y)))
    return X, y

def test(feature_name, testset_index):
    """Test for a test set from ./train_test folder with specific feature_name
    
    Arguments:
        feature_name {string} -- feature_name (Ex. 'keras_vgg16_fc2').
        testset_index {int} -- index of test file in ./train_set folder (Ex. 0).
    """
    logger = logging.getLogger()
    logger.info('Start testing for test set {} with feature {}...'.format(testset_index, feature_name))
    models_dir = MODELS_DIR_FORMAT.format(feature_name, testset_index)
    if not os.path.exists(models_dir):
        logger.error('!Error: Models directory {} not found'.format(models_dir))
    X, y = get_test_data(feature_name, testset_index)
    accuracy_file = '{}/accuracy.csv'.format(ACCURAY_FILE_FORMAT.format(feature_name, testset_index))
    if OVERWRITE_MODE:
        with open(accuracy_file, 'wb') as f:
            f.write('')
    for model_file in glob.glob('{}/*.pkl'.format(models_dir)):
        model = joblib.load(model_file) 
        file_name = os.path.basename(model_file)
        model_name = file_name.replace('.pkl','')
        test_result_folder = TEST_RESULT_FOLDER_FORMAT.format(feature_name, testset_index, model_name)
        confusion_matrix_file = '{}/confusion_matrix.csv'.format(test_result_folder)
        miss_samples_file = '{}/miss_samples.csv'.format(test_result_folder)
        if (not OVERWRITE_MODE) and os.path.exists(confusion_matrix_file) and os.path.exists(miss_samples_file):
            logger.info('Test result for {} existed'.format(test_result_folder))
            continue
        if not os.path.exists(test_result_folder):
            logger.info('Creating test result folder {}...'.format(test_result_folder))
            os.makedirs(test_result_folder)
        logger.info('Predicting with model: {}'.format(model_name))
        y_pred = model.predict(X)
        accuracy = tools.calculate_accuracy(y, y_pred)
        logger.info('Done, accuracy = {}'.format(accuracy))
        with open(accuracy_file, 'a') as fw:
            fw.write('{},{}\n'.format(model_name, accuracy))
        cf_matrix = confusion_matrix(y, y_pred, np.unique(y))
        logger.info('Writing confusion matrix to {}...'.format(confusion_matrix_file))
        with open(confusion_matrix_file, 'w') as f:
            f.write(np.array2string(cf_matrix, separator=', '))
        logger.info('Writting miss samples to {}...'.format(miss_samples_file))
        with open(miss_samples_file, 'w') as f:
            f.write('index,y,y_pred,missed\n')
            for i in range(0, len(y)):
                if y[i] == y_pred[i]:
                    f.write('{},{},{},\n'.format(i, y[i], y_pred[i]))
                else:
                    f.write('{},{},{},miss\n'.format(i, y[i], y_pred[i]))
        

def test_for(feature_name):
    """Test for all test set in ./train_test folder with specific feature name
    
    Arguments:
        feature_name {string} -- feature name (Ex. 'keras_vgg16_fc2')
    """
    logger = logging.getLogger()
    logger.info('Start testing for feature {}...'.format(feature_name))
    feature_dir = FEATURES_DIR_FORMAT.format(feature_name)
    if not os.path.exists(feature_dir):
        logger.error('!Error: Feature folder {} not found. Please run feature extractor for this feature'.format(feature_dir))
        return
    for i in range(LOW_TESTSET_INDEX, HIGH_TESTSET_INDEX):
        logger.info('>'*100)
        test(feature_name, i)
        logger.info('>'*100)

def main(argv):
    tools.config()
    if OVERWRITE_MODE:
        with open(LOG_FILE, 'wb') as fw:
            fw.write('')
    logger = tools.get_logger(LOG_FILE, logging.INFO, logging.DEBUG)
    if len(argv) == 1:
        logger.debug('Try again with feature name args. Ex. `python test.py keras_vgg16_fc2 alexnet`')
        return
    for i in range(1, len(argv)):
        logger.info('*'*100)
        feature_name = argv[i]
        test_for(feature_name)
        logger.info('*'*100)

if __name__ == '__main__':
    main(sys.argv)