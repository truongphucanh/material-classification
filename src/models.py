# models, lk, lc, ld, lg, lp = models.getmodels('../config/models_config.csv')
from sklearn import svm
import numpy
import os
import csv

def get_models(config_file='../config/models_config.csv'):
    if not config_file.endswith('.csv'):
        print('!Error: models config file must be *.csv, but it is {}'.format(config_file))
        return
    if not os.path.exists(config_file):
        print('!Error: File not found {}'.format(config_file))
        return
    models = []
    list_kernel = []
    list_C = []
    list_degree = []
    list_gamma = []
    with open(config_file) as f:
        reader = csv.reader(f)
        rows = list(reader)
        isFirstRow = True
        for row in rows:
            if len(row) < 4:
                print('!Error: Invalid format in models config file {}. Please folow format \"kernel, C, degree, gamma\"'.format(config_file))
                return
            if isFirstRow:
                print('Skip first row')
                isFirstRow = False
                continue
            kernel = row[0]
            C = row[1]
            degree = row[2]
            gamma = row[3]
            model = svm.LinearSVC()
            if kernel == 'default':
                model = svm.SVC(probability=True)
            elif kernel == 'linear':
                model = svm.SVC(kernel=kernel, C=float(C), probability=True)
            elif kernel == 'poly':
                model = svm.SVC(kernel=kernel, C=float(C), degree=float(degree), probability=True)
            elif kernel == 'rbf':
                model = svm.SVC(kernel=kernel, C=float(C), gamma=float(gamma), probability=True)
            else:
                print('!Error: Invalid kernel. Kernel must be one of `linear, rbf or poly` but it is {}').format(kernel)
                continue
            models.append(model)
            list_kernel.append(kernel)
            list_C.append(C)
            list_degree.append(degree)
            list_gamma.append(gamma)
    return numpy.array(models), numpy.array(list_kernel), numpy.array(list_C), numpy.array(list_degree), numpy.array(list_gamma)