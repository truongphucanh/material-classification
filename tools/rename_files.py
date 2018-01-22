import os
import glob

def get_labels():
    with open('../train_test/classInd.txt') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    labels = [line.split()[1] for line in content]
    return labels

def main():
    labels = get_labels()
    for label in labels:
        features_folder = '../bin/features/keras_vgg16_fc2/{}'.format(label)
        for file in glob.glob('{}/*.pkl'.format(features_folder)):
            new_name = file.replace('_X','')
            os.rename(file, new_name)
        labels_folder = '../bin/labels/{}'.format(label)
        for file in glob.glob('{}/*.pkl'.format(labels_folder)):
            new_name = file.replace('_y','')
            os.rename(file, new_name)

if __name__ == '__main__':
    main()