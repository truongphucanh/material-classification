# Format: python combine_prediction.py <dataset_name> <split name> <feature_1 name> <model_1 name> <feature_2 name> <model_2 name> ... <feature_i name> <model_i name>
# Example: python combine_prediction.py FMD test_split_0 original default_default_default_default original+edges default_default_default_default
from mkit import mlog
from mkit import mlearning
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import dataset
import sys
import pickle
import os
import logging
import numpy

def combine(lprobs):
    if len(lprobs) <= 0:
        return
    ret = lprobs[0]
    for i in range(1, len(lprobs)):
        ret = ret + lprobs[i]
    return ret

def parse_argv(argv):
    """Parse command line arguments
    
    Arguments:
        argv {list} -- sys.argv
    
    Returns:
        string, list, list -- Splitname, feature names, model names
    """
    if len(argv) < 6:
        return
    dataset_name = argv[1]
    split = argv[2]
    features = []
    models = []
    i = 3
    while i < len(argv):
        features.append(argv[i])
        models.append(argv[i + 1])
        i = i + 2
    return dataset_name, split, features, models

def get_list_probs(dataset_name, split, features, models):
    logger = logging.getLogger()
    lprobs = []
    for i in range (0, len(features)):
        pkl = "./DS_{}/result/test/{}/{}/{}/probs.pkl".format(dataset_name, features[i], split, models[i])
        if not os.path.exists(pkl):
            logger.error("!Error. Find {} not found.".format(pkl))
            return []
        with open(pkl, "rb") as f:
            probs = pickle.load(f)
            lprobs.append(probs)
    return lprobs

def get_result_folder(dataset_name, split, features, models):
    feature_model = ""
    for i in range(0, len(features) - 1):
        feature_model = feature_model + features[i] + "#" + models[i] + "~"
    feature_model = feature_model + features[len(features) - 1] + "#" + models[len(models) - 1]
    result_folder = "./DS_{}/result/test/combine_prediction/{}".format(dataset_name,feature_model)
    return result_folder

def get_classes(dataset_name, split, feature, model):
    clf = joblib.load("./DS_{}/result/models/{}/{}/{}.pkl".format(dataset_name, feature, split.replace("test", "train"), model))
    return clf.classes_

def main(argv):
    # config and get logger
    mlog.config()
    logger = mlog.get_logger("./logs/combine_prediction.log")

    # parse argv
    dataset_name, split, features, models = parse_argv(argv)

    # get file names of result
    result_folder = get_result_folder(dataset_name, split, features, models)
    accuracy_file = "{}/accuracy.csv".format(result_folder)
    confusion_matrix_file = "{}/{}/confusion_matrix.csv".format(result_folder, split)
    miss_samples_file = "{}/{}/miss_samples.csv".format(result_folder, split)
    
    # if result existed, do nothing
    if os.path.exists(confusion_matrix_file) and os.path.exists(miss_samples_file):
        logger.debug("Result existed. Do nothing")
        return

    # get list of probality result from file
    lprobs = get_list_probs(dataset_name, split, features, models)

    # combine probability prediction
    combined_probs = combine(lprobs)

    # get labels of combined_probs
    y_pred = mlearning.get_labels(combined_probs, get_classes(dataset_name, split, features[0], models[0]))

    # get expected labels
    y = dataset.get_y(dataset_name, split)

    # create result folder
    if not os.path.exists("{}/{}".format(result_folder, split)):
        os.makedirs("{}/{}".format(result_folder, split))

    

    # accuracy
    accuracy = accuracy_score(y, y_pred)
    logger.debug("accuracy = {}".format(accuracy))
    with open(accuracy_file, "a") as fw:
        fw.write("{},{}\n".format(split, accuracy))
    
    # confusion matrix
    cf_matrix = confusion_matrix(y, y_pred, numpy.unique(y))
    logger.debug("Writing confusion matrix to {}...".format(confusion_matrix_file))
    with open(confusion_matrix_file, "w") as f:
        f.write(numpy.array2string(cf_matrix, separator=", "))
    
    # missed samples
    logger.debug("Writting miss samples to {}...".format(miss_samples_file))
    with open(miss_samples_file, "w") as f:
        f.write("index,y,y_pred,missed\n")
        for i in range(0, len(y)):
            if y[i] == y_pred[i]:
                f.write("{},{},{},\n".format(i, y[i], y_pred[i]))
            else:
                f.write("{},{},{},miss\n".format(i, y[i], y_pred[i]))
    return 0

if __name__ == "__main__":
    main(sys.argv)
