"""Get labels for splits set."""

import os
import pickle
import logging
from mkit import mlog

TRAIN_FORMAT = 'trainlist0{}'
TEST_FORMAT = 'testlist0{}'

def get_labels(train_test_file):
    logger = logging.getLogger()
    src_file = '../splits/{}.txt'.format(train_test_file)
    labels_file = '../bin/labels/{}.pkl'.format(train_test_file)
    if os.path.exists(labels_file):
        logger.debug('Labels file {} existed.'.format(labels_file))
        return
    if not os.path.exists(src_file):
        logger.error('!Error: Find {} not found'.format(src_file))
        return
    logger.debug('Running on {}...'.format(src_file))
    labels = []
    with open(src_file) as fr:
        content = fr.readlines()
    for line in content:
        label = line.split()[1]
        count = line.split()[2]
        labels.extend([label] * int(count))
    logger.info(labels)
    with open(labels_file, 'wb') as fw:
        pickle.dump(labels, fw)

def main():
    mlog.get_logger('../logs/get_labels.log', logging.INFO, logging.DEBUG)
    for i in range(0, 6):
        train_file = TRAIN_FORMAT.format(i)
        test_file = TEST_FORMAT.format(i)
        get_labels(train_file)
        get_labels(test_file)

if __name__ == '__main__':
    main()