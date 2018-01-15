import os

def main():
    with open('../train_test/classInd.txt') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    labels = [line.split()[1] for line in content]
    for label in labels:
        directory = '../data/{}'.format(label)
        if not os.path.exists(directory):
            print 'Creating folder {}...'.format(directory)
            os.makedirs(directory)
        else:
            print 'Folder {} existed.'.format(directory)

        directory = '../bin/train_test/{}'.format(label)
        if not os.path.exists(directory):
            print 'Creating folder {}...'.format(directory)
            os.makedirs(directory)
        else:
            print 'Folder {} existed.'.format(directory)

if __name__ == '__main__':
    main()
