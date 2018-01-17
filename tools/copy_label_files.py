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
        from_folder = '../bin/features/keras_vgg16_fc2/{}'.format(label)
        to_folder = '../bin/labels/{}'.format(label)
        for from_dir in glob.glob('{}/*_y.pkl'.format(from_folder)):
            file_name_only = os.path.basename(from_dir)
            to_dir = '{}/{}'.format(to_folder, file_name_only)
            if os.path.exists(to_dir):
                print 'File {} existed.'.format(to_dir)
            else:
                print 'Moving file from {} to {}'.format(from_dir, to_dir)
                os.rename(from_dir, to_dir)

if __name__ == '__main__':
    main()