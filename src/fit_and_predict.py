from sklearn import svm, metrics
from sklearn.externals import joblib
import numpy as np
import pickle as pkl
import time
import logging
import tools
import cPickle
import time
import os
import csv

IS_TESTING = True
FEATURES_DIR = '../bin/features'
MODELS_CONFIG = '../config/models_config.csv'

def get_data(train_list_file, feature):
    """Get features, labels for training process.

    Parameters
    ----------
    train_list_file: string
        One of file names in ../train_test/.
    feature: string
        Feature name.

    Returns
    -------
    X: 2D-list
        List of feature vectors with shape (n_samples, 4096).
    y: 1D-list
        List of labels with shape (n_samples).
    folders: 1D-list
        List of folder which X was extracted from.

    Example
    -------
    X, y, folders = get_data('../train_set/trainlist01', 'keras_vgg16_fc2').

    """
    with open(train_list_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    X = []
    y = []
    folders = []
    for line in content:
        folder = line.split()[0]
        label = line.split()[1]
        curr_X = []
        curr_y = []
        curr_folder = []
        feature_file = '{}/{}/{}.pkl'.format(FEATURES_DIR, feature, folder)
        if not os.path.exists(feature_file):
            print '!Error: feature file {} not existed.'.format(feature_file)
            continue
        with open(feature_file, 'rb') as f:
            curr_X = pkl.load(f)
            curr_y = [label] * len(curr_X)
            curr_folder = [folder] * len(curr_X)
        X.extend(curr_X)
        y.extend(curr_y)
        folders.extend(curr_folder)
    return X, y, folders

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
    logger = tools.get_logger(file_name='', file_level=None, console_level=logging.DEBUG)
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
    with open(models_config_file, 'rb') as f:
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
            if kernel == 'linear':
                model = svm.SVC(kernel=kernel, C=float(C))
            elif kernel == 'rbf':
                model = svm.SVC(kernel=kernel, C=float(C), gamma=float(gamma))
            elif kernel == 'poly':
                model = svm.SVC(kernel=kernel, C=float(C), degree=float(degree))
            else:
                logger.error('!Error: Invalid kernel. Kernel must be one of `linear, rbf or poly` but it is {}').format(kernel)
            models.append(model)
            list_kernel.append(kernel)
            list_C.append(C)
            list_degree.append(degree)
            list_gamma.append(gamma)
    return models, list_kernel, list_C, list_degree, list_gamma
    

def train_and_test(train_list_file, test_list_file, idx=0):
    # 1. get data
    X_train, y_train, _ = get_data(train_list_file)
    X_test, y_test, _ = get_data(test_list_file)

    # 2. config parameter of model
    C = 0.1
    kernel = 'linear'
    gamma = None
    degree = None
    multi_class = 'ovr'
    
    # 3. get logger
    log_file = '../logs/vgg16-svm/train_test_{}.log'.format(idx)
    logger = tools.get_logger(log_file, logging.DEBUG, logging.DEBUG)
    logger.info('*'*75)
    logger.info('C: {}'.format(C))
    logger.info('multi_class: {}'.format(multi_class))
    logger.info('kernel: {}'.format(kernel))
    logger.info('gamma: {}'.format(gamma))
    logger.info('degree: {}'.format(degree))

    # 4. fit data
    model = svm.LinearSVC(C=C)
    start_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_time
    logger.info('fit_time: {} seconds'.format(fit_time))
    
    # 5. save model
    model_file = '../bin/vgg16-svm/models/train_test_{}/{}_{}_{}_{}_{}.pkl'.format(idx, multi_class, C, kernel, gamma, degree)
    with open(model_file, 'wb') as fid:
        joblib.dump(model, model_file)
    
    # 6. predict on test data
    y_pred_test = model.predict(X_test)
    
    # 7. save result
    mis_indecies, accuracy = tools.calculate_accuracy(y_test, y_pred_test)
    logger.info('accuracy: {}'.format(accuracy))
    logger.info('missed indecies samples:\n{}'.format(mis_indecies))
    #logger.info('score on test set: {}'.format(model.score(X_test, y_test)))
    logger.info('confusion_matrix:\n {}'.format(metrics.confusion_matrix(y_test, y_pred_test, np.unique(y_test))))
    logger.info('%-15s\t%-15s' % ('y_test', 'y_pred_test'))
    for idx in range(len(y_test)):
        logger.info('%-15s\t%-15s' % (y_test[idx], y_pred_test[idx]))

def test():
    info_train_file = '../train_test/trainlist00.txt'
    info_test_file = '../train_test/testlist00.txt'
    #train_and_test(info_train_file, info_test_file, 0)
    models, list_kernel, list_C, list_degree, list_gamma = get_models(MODELS_CONFIG)
    print models

def main():
    tools.config()
    # model_file = '../bin/vgg16-svm/models/train_test_0/ovr_0.1_linear_None_None.pkl'
    # with open(model_file, 'rb') as fid:
    #     model = joblib.load(model_file) 
    #     info_test_file = '../train_test/testlist01-copy.txt'
    #     X_test, y_test = get_data(info_test_file)
    #     y_pred_test = model.predict(X_test)
    #     mis_indecies, accuracy = tools.calculate_accuracy(y_test, y_pred_test)
    #     print accuracy
    #     return 0
    if IS_TESTING:
        test()
        return 0
    N_FILES = 5
    for i in range (1, N_FILES + 1):
        info_train_file = '../train_test/trainlist0{}.txt'.format(str(i))
        info_test_file = '../train_test/testlist0{}.txt'.format(str(i))
        train_and_test(info_train_file, info_test_file)
    return 0

if __name__ == '__main__':
    main()