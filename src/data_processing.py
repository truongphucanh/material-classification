def get_models(models_config_file):
    """Get sklearn.svm.SVC models
    
    Arguments:
        models_config_file {string} -- Model config file name (*.csv)
    
    Returns:
        models -- List of models
    """
    logger = logging.getLogger()
    if not models_config_file.endswith('.csv'):
        logger.error('!Error: models config file must be *.csv, but it is {}'.format(models_config_file))
        return
    if not os.path.exists(models_config_file):
        logger.error('!Error: File not found {}'.format(models_config_file))
        return
    models = []
    list_kernel = []
    list_C = []
    list_degree = []
    list_gamma = []
    list_prob = []
    with open(models_config_file) as f:
        reader = csv.reader(f)
        rows = list(reader)
        isFirstRow = True
        for row in rows:
            if len(row) < 5:
                logger.error('!Error: Invalid format in models config file {}. Please folow format \"kernel, C, degree, gamma\"'.format(models_config_file))
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
    return models, list_kernel, list_C, list_degree, list_gamma

def get_X(split_file, feature_name):
    """Get X for SVMs train or test
    
    Arguments:
        split_file {string} -- Split file name
        feature_name {string} -- Feature name
    
    Returns:
        list -- List of feature vector
    """
    logger = logging.getLogger()
    with open(split_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    X = np.array()
    for line in content:
        folder = line.split()[0]
        feature_folder = '../bin/features/{}/{}'.format(feature_name, folder)
        logger.debug('Run on feature folder: {}'.format(feature_folder))
        for pkl_file in glob.glob('{}/*.pkl'.format(feature_folder)):
            with open(pkl_file, 'rb') as f:
                X = np.append(X, pickle.load(f))
    return X

def get_train_data(feature_name, trainset_index):
    """Get training data
    
    Arguments:
        feature_name {string} -- feature name (Ex. 'keras_vgg16_fc2')
        trainset_index {int} -- index of training set in folder `./train_test/`
    
    Returns:
        X, y -- List of features (X) and labels (y)
    """
    logger = logging.getLogger()
    logger.info('Getting training data for training set {} with feature {}...'.format(trainset_index, feature_name))
    X_file = X_FILE_FORMAT.format(feature_name, trainset_index)
    y_file = Y_FILE_FORMAT.format(trainset_index)
    X = []
    y = []
    if not os.path.exists(X_file):
        
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