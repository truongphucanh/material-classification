# models, lk, lc, ld, lg, lp = models.getmodels('../config/models_config.csv')
from sklearn import svm
import numpy
import os
import csv
import kit

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
    list_prob = []
    with open(config_file) as f:
        reader = csv.reader(f)
        rows = list(reader)
        isFirstRow = True
        for row in rows:
            if len(row) < 5:
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
                print('!Error: Invalid kernel. Kernel must be one of `linear, rbf or poly` but it is {}').format(kernel)
            models.append(model)
            list_kernel.append(kernel)
            list_C.append(C)
            list_degree.append(degree)
            list_gamma.append(gamma)
            list_prob.append(prob)
    return numpy.array(models), numpy.array(list_kernel), numpy.array(list_C), numpy.array(list_degree), numpy.array(list_gamma), numpy.array(list_prob)

def confirm():
    models, lk, lc, ld, lg, lp = get_models('../config/models_config.csv')
    print(lk, lc, ld, lg, lp)

# confirm()