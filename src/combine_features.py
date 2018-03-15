# python ./src/combine_features.py <dataset_name> <feature_1> <feature_2> <feature_3> ...
# python ./src/combine_features.py FMD original edges texture

import sys
import numpy as np
import glob
import os
import pickle

def main(argv):
    if len(argv) < 4:
        print('Missing arguments: python ./src/combine_features.py <dataset_name> <feature_1> <feature_2> <feature_3> ...')
        return

    dataset_name = argv[1]

    folder_feature1 = './DS_{}/result/features/{}'.format(dataset_name, argv[2])
    combine_feature_name = ''

    for i in range(2, len(argv) - 1):
        combine_feature_name = combine_feature_name + argv[i] + '+'
    combine_feature_name = combine_feature_name + argv[len(argv) - 1]
    print('Combined-feature name: {}'.format(combine_feature_name))

    for dirpath, _, filenames in os.walk(folder_feature1):
        for filename in [f for f in filenames if f.endswith(".pkl")]:
            if ('train' in filename) or ('test' in filename):
                print('skip train test')
                continue
            feature1_pkl_path = os.path.join(dirpath, filename).replace("\\","/")
            combined_feature_path = feature1_pkl_path.replace(argv[2], combine_feature_name)
            combined_feature_folder = dirpath.replace(argv[2], combine_feature_name)
            if os.path.exists(combined_feature_path):
                print('Combined-feature file {} existed.'.format(combined_feature_path))
            else:
                if not os.path.exists(combined_feature_folder):
                    os.makedirs(combined_feature_folder)
                    print('Running on folder {}'.format(combined_feature_folder))
                combined_feature = []
                for i in range(2, len(argv)):
                    feature_path = feature1_pkl_path.replace(argv[2], argv[i])
                    print('Extend with feature from {}'.format(feature_path))
                    with open(feature_path, 'rb') as f:
                        combined_feature.extend(pickle.load(f))
                print('Extended. Shape {}. Writting combined feature to file {}'.format(np.shape(combined_feature),combined_feature_path))
                with open(combined_feature_path, 'wb') as f:
                    pickle.dump(combined_feature, f)

    print('Done')
    return 0

if __name__ == '__main__':
    main(sys.argv)