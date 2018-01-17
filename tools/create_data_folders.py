import os

def make_folders(directory, labels):
    print 'Making labels folders in {}'.format(directory)
    for label in labels:
        new_folder = '{}/{}'.format(directory,label)
        if not os.path.exists(new_folder):
            print 'Creating folder {}...'.format(new_folder)
            os.makedirs(new_folder)
        else:
            print 'Folder {} existed.'.format(new_folder)

def get_labels():
    with open('../train_test/classInd.txt') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    labels = [line.split()[1] for line in content]
    return labels

def main():
    labels = get_labels()
    data_directory = '../data'
    make_folders(data_directory, labels)
    labels_directory = '../bin/labels'
    make_folders(labels_directory, labels)

if __name__ == '__main__':
    main()
