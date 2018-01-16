from sklearn import svm, metrics
import numpy as np
import pickle as pkl
import time
import logging
import tools
import cPickle
import time

def get_data(train_list_file):
    with open(train_list_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    X = []
    y = []
    for line in content:
        folder = line.split()[0]
        with open('../bin/train_test/{}_X.pkl'.format(folder)) as f:
            X.extend(pkl.load(f))
        with open('../bin/train_test/{}_y.pkl'.format(folder)) as f:
            y.extend(pkl.load(f))
    return X, y

def train_and_test(train_list_file, test_list_file, idx=0):
    # 1. get data
    X_train, y_train = get_data(train_list_file)
    X_test, y_test = get_data(test_list_file)

    # 2. config parameter of model
    C = 0.1
    kernel = 'linear'
    gamma = None
    degree = None
    multi_class = 'ovr'
    
    # 3. get logger
    log_file = '../logs/svm/train_test_{}.log'.format(idx)
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
    model_file = '../bin/models/svm/train_test_{}.pkl'.format(idx)
    with open(model_file, 'wb') as fid:
        cPickle.dump(model, fid)
    
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
    info_train_file = '../train_test/trainlist01-copy.txt'
    info_test_file = '../train_test/testlist01-copy.txt'
    train_and_test(info_train_file, info_test_file, 0)

IS_TESTING = True

def main():
    tools.config()
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