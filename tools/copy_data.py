import os
from distutils.dir_util import copy_tree

def run(info_file):
    with open(info_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    names = [line.split()[0] for line in content]
    for name in names:
        src = '../data/GTOS_256/{}'.format(name.split('/')[1])
        dest = '../data/{}'.format(name)
        if not os.path.exists(dest):
            os.makedirs(dest)
            print 'Copying from {} to {}...'.format(src, dest)
            copy_tree(src, dest)
        else:
            print 'Folder {} existed.'.format(dest)

def main():
    if not os.path.exists('../data/GTOS_256'):
        print 'Please download dataset GTOS_256, extract it to ../data and run the script again.'
        return -1
    
    run('../train_test/testlist01.txt')
    run('../train_test/testlist02.txt')
    run('../train_test/testlist03.txt')
    run('../train_test/testlist04.txt')
    run('../train_test/testlist05.txt')

    run('../train_test/trainlist01.txt')
    run('../train_test/trainlist02.txt')
    run('../train_test/trainlist03.txt')
    run('../train_test/trainlist04.txt')
    run('../train_test/trainlist05.txt')

if __name__ == '__main__':
    main()
