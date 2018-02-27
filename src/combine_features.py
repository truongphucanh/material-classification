"""Combine features to get another feature (lager feature vector)"""
import sys
import numpy as np
import glob
import tools
import logging
import os
import pickle

def main(argv):
    tools.config()
    if len(argv) < 3:
        print('Missing arguments: combine_features.py `feature_name1` `feature_name2`. \
            Ex. `python combine_features.py keras_vgg16_fc2-original keras_vgg16_fc2-edges`')
        return

    log_file = '../logs/combine-features.log'
    logger = tools.get_logger(log_file, logging.INFO, logging.DEBUG)
    logger.info('*' * 100)

    folder_feature1 = '../bin/features/{}'.format(argv[1])
    combine_feature_name = ''

    for i in range(1, len(argv) - 1):
        combine_feature_name = combine_feature_name + argv[i] + '+'
    combine_feature_name = combine_feature_name + argv[len(argv) - 1]
    logger.info('Combined-feature name: {}'.format(combine_feature_name))
    logger.debug(folder_feature1)

    for dirpath, _, filenames in os.walk(folder_feature1):
        for filename in [f for f in filenames if f.endswith(".pkl")]:
            if ('train' in filename) or ('test' in filename):
                logger.debug('skip train test')
                continue
            feature1_pkl_path = os.path.join(dirpath, filename)
            combined_feature_path = feature1_pkl_path.replace(argv[1], combine_feature_name)
            combined_feature_folder = dirpath.replace(argv[1], combine_feature_name)
            if os.path.exists(combined_feature_path):
                logger.debug('Combined-feature file {} existed.'.format(combined_feature_path))
            else:
                if not os.path.exists(combined_feature_folder):
                    os.makedirs(combined_feature_folder)
                    logger.info('Running on folder {}'.format(combined_feature_folder))
                combined_feature = []
                for i in range(1, len(argv)):
                    feature_path = feature1_pkl_path.replace(argv[1], argv[i])
                    logger.debug('Extend with feature from {}'.format(feature_path))
                    with open(feature_path, 'rb') as f:
                        combined_feature.extend(pickle.load(f))
                logger.debug('Extended. Shape {}. Writting combined feature to file {}'.format(np.shape(combined_feature),combined_feature_path))
                with open(combined_feature_path, 'wb') as f:
                    pickle.dump(combined_feature, f)

    logger.debug('Done')
    return 0

if __name__ == '__main__':
    main(sys.argv)