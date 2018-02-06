# Example python test.py keras_vgg16_fc2-original 1 5
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import glob
import pickle
import sys
import time
import logging
import os
import csv
import numpy
import dataset
from mkit import mlog
from mkit import mlearning

OVERWRITE = False

def test(feature_name, split_name):
    """Test for a test set from ./train_test folder with specific feature_name
    
    Arguments:
        feature_name {string} -- feature_name (Ex. 'keras_vgg16_fc2-original').
        split_name {string} -- split name(Ex. 'testlist01').
    """
    # get logger
    logger = logging.getLogger()
    logger.info('Start testing for '.format(feature_name, split_name))

    # if there is no model has been trained for this split, just return
    models_dir = '../bin/models/{}/{}'.format(feature_name, split_name).replace('testlist', 'trainlist')
    if not os.path.exists(models_dir):
        logger.error('!Error: Models directory {} not found'.format(models_dir))
        return
    
    # get data
    X, y = dataset.get_data(feature_name, split_name)

    # clean up accuracy file if in overwrite mode
    accuracy_file = '{}/accuracy.csv'.format('../bin/test/{}/{}'.format(feature_name, split_name))
    if OVERWRITE:
        with open(accuracy_file, 'wb') as f:
            f.write('')
    
    # get trained model to predict
    for model_file in glob.glob('{}/*.pkl'.format(models_dir)):
        # load trained model
        model = joblib.load(model_file)

        # get model name to write result
        file_name = os.path.basename(model_file)
        model_name = file_name.replace('.pkl','')

        # get folder, files name to write result
        test_result_folder = '../bin/test/{}/{}/{}'.format(feature_name, split_name, model_name)
        confusion_matrix_file = '{}/confusion_matrix.csv'.format(test_result_folder)
        miss_samples_file = '{}/miss_samples.csv'.format(test_result_folder)
        probs_file = '{}/probs.pkl'.format(test_result_folder)

        # if not overwrite mode and result for this split exsited, just skip
        if (not OVERWRITE) and os.path.exists(confusion_matrix_file) and os.path.exists(miss_samples_file):
            logger.info('Test result for {} existed'.format(test_result_folder))
            continue
        
        # creating folder for result if it not existed
        if not os.path.exists(test_result_folder):
            logger.info('Creating test result folder {}...'.format(test_result_folder))
            os.makedirs(test_result_folder)
        
        # predict probability
        logger.info('Predicting with model: {}'.format(model_name))
        probs =  model.predict_proba(X)
        
        # saving probability result
        with open(probs_file, 'wb') as f:
            pickle.dump(probs, f)

        # get labels
        y_pred = mlearning.get_labels(probs, model.classes_)
        
        # accuracy
        accuracy = accuracy_score(y, y_pred)
        logger.info('Done, accuracy = {}'.format(accuracy))
        with open(accuracy_file, 'a') as fw:
            fw.write('{},{}\n'.format(model_name, accuracy))
        
        # confusion matrix
        cf_matrix = confusion_matrix(y, y_pred, numpy.unique(y))
        logger.info('Writing confusion matrix to {}...'.format(confusion_matrix_file))
        with open(confusion_matrix_file, 'w') as f:
            f.write(numpy.array2string(cf_matrix, separator=', '))
        
        # missed samples
        logger.info('Writting miss samples to {}...'.format(miss_samples_file))
        with open(miss_samples_file, 'w') as f:
            f.write('index,y,y_pred,missed\n')
            for i in range(0, len(y)):
                if y[i] == y_pred[i]:
                    f.write('{},{},{},\n'.format(i, y[i], y_pred[i]))
                else:
                    f.write('{},{},{},miss\n'.format(i, y[i], y_pred[i]))
        
def test_for(feature_name, low, high):
    # get logger
    logger = logging.getLogger()
    logger.info('Start testing for feature {}...'.format(feature_name))

    # test for splits
    for i in range(low, high + 1):
        logger.info('>'*100)
        test(feature_name, 'testlist0{}'.format(i))

def main(argv):
    mlog.config()
    logger = mlog.get_logger('../logs/test.log')
    if len(argv) <= 1:
        print('Missing arguments. Please try again.')
        print('Format: python test.py \{feature name\}')
        print('Example: python test.py keras_vgg16_fc2-original')
        return
    feature_name = argv[1]
    LOW_INDEX = 1
    HIGH_INDEX = 5
    if len(argv) >= 4:
        LOW_INDEX = int(argv[2])
        HIGH_INDEX = int(argv[3])
    test_for(feature_name, LOW_INDEX, HIGH_INDEX)

if __name__ == '__main__':
    main(sys.argv)