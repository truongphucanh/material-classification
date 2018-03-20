# Format:   python ./src/train.py <dataset_name> <feature_type> <trainsplit_index_low> <trainsplit_index_high>
# Example:  python ./src/train.py FMD original 1 1

from sklearn.externals import joblib
import pickle
import sys
import time
import logging
import os
import numpy
import models
import dataset
import glob
from mkit import mlog

OVERWRITE = False

def train(dataset_name, feature_type, split_name):
    # get logger
    logger = logging.getLogger()
    logger.info("Start training: DS_{}, feature {}, split {}".format(dataset_name, feature_type, split_name))

    # get models
    clfs, lkernel, lC, ld, lg = models.get_models("./config/models_config.csv")

    # get data
    feature_dir = "./DS_{}/result/features/{}".format(dataset_name, feature_type)
    if not os.path.exists(feature_dir):
        logger.error("!Error: Feature folder {} not found".format(feature_dir))
        return
    X, y = dataset.get_data(dataset_name, feature_type, split_name)

    # fit data and save model to file
    models_dir = "./DS_{}/result/models/{}/{}".format(dataset_name, feature_type, split_name)
    fittime_file = "{}/time.csv".format(models_dir)
    if not os.path.exists(models_dir):
        logger.info("Creating models folder {}...".format(models_dir))
        os.makedirs(models_dir)
    if OVERWRITE:
        with open(fittime_file, "wb") as fw:
            fw.write("Kernel,C,degree,gamma,fit time\n")
    for i, clf in enumerate(clfs):
        model_file = "{}_{}_{}_{}.pkl".format(lkernel[i], lC[i], ld[i], lg[i])
        model_dir = "{}/{}".format(models_dir, model_file)
        if (not OVERWRITE) and os.path.exists(model_dir):
            logger.info("Model {} existed.".format(model_dir))
            continue
        logger.info("Fitting with model {}: {}".format(i, model_file))
        start_time = time.time()
        clf.fit(X, y)
        fit_time = time.time() - start_time
        logger.info("Done. fit_time = {}".format(fit_time))
        logger.info("Saving model {} to {} ...".format(i, models_dir))
        joblib.dump(clf, model_dir)
        with open(fittime_file, "a") as fw:
            fw.write("{},{},{},{},{}\n".format(lkernel[i], lC[i], ld[i], lg[i], fit_time))

def train_for(dataset_name, feature_type, low, high):
    logger = logging.getLogger()
    feature_dir = "./DS_{}/result/features/{}".format(dataset_name, feature_type)
    if not os.path.exists(feature_dir):
        logger.error("!Error: Feature folder {} not found".format(feature_dir))
        return
    for i in range(low, high + 1):
        train(dataset_name, feature_type, "train_split_{}".format(i))

def main(argv):
    mlog.config()
    logger = mlog.get_logger("./logs/train.log")
    if len(argv) < 3:
        print("Missing arguments. Please try again.")
        print("# Format:   python ./src/train.py <dataset_name> <feature_type> <trainsplit_index_low> <trainsplit_index_high>")
        print("# Example:  python ./src/train.py FMD original 1 1")
        return
    dataset_name = argv[1]
    feature_type = argv[2]
    LOW_INDEX = 1
    HIGH_INDEX = 5
    if len(argv) >= 4:
        LOW_INDEX = int(argv[3])
        HIGH_INDEX = int(argv[4])
    train_for(dataset_name, feature_type, LOW_INDEX, HIGH_INDEX)

if __name__ == "__main__":
    main(sys.argv)
