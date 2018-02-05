# models, lk, lc, ld, lg, lp = models.getmodels('../config/models_config.csv')
import numpy
import os
import csv
import logging
import kit
from sklearn import svm

def get_models(config_file):
    logger = logging.getLogger()
    if not config_file.endswith('.csv'):
        logger.error('!Error: models config file must be *.csv, but it is {}'.format(config_file))
        return
    if not os.path.exists(config_file):
        logger.error('!Error: File not found {}'.format(config_file))
        return
    models = []
    list_kernel = []
    list_C = []
    list_degree = []
    list_gamma = []
    list_prob = []
    with open(config_file) as f:
        reader = csv.reader(f)
        rows = list(reader)
        isFirstRow = True
        for row in rows:
            if len(row) < 5:
                logger.error('!Error: Invalid format in models config file {}. Please folow format \"kernel, C, degree, gamma\"'.format(config_file))
                return
            if isFirstRow:
                logger.debug('Skip first row')
                isFirstRow = False
                continue
            kernel = row[0]
            C = row[1]
            degree = row[2]
            gamma = row[3]
            prob = row[4]
            model = svm.LinearSVC()
            if kernel == 'default':
                model = svm.SVC(probability=kit.str2bool(prob))
            elif kernel == 'linear':
                model = svm.SVC(kernel=kernel, C=float(C), probability=kit.str2bool(prob))
            elif kernel == 'poly':
                model = svm.SVC(kernel=kernel, C=float(C), degree=float(degree), probability=kit.str2bool(prob))
            elif kernel == 'rbf':
                model = svm.SVC(kernel=kernel, C=float(C), gamma=float(gamma), probability=kit.str2bool(prob))
            else:
                logger.error('!Error: Invalid kernel. Kernel must be one of `linear, rbf or poly` but it is {}').format(kernel)
            models.append(model)
            list_kernel.append(kernel)
            list_C.append(C)
            list_degree.append(degree)
            list_gamma.append(gamma)
            list_prob.append(prob)
    return numpy.array(models), numpy.array(list_kernel), numpy.array(list_C), numpy.array(list_degree), numpy.array(list_gamma), numpy.array(list_prob)

def confirm():
    logger = kit.get_logger('../logs/models.log')
    models, lk, lc, ld, lg, lp = get_models('../config/models_config.csv')
    print(lk, lc, ld, lg, lp)

# confirm()