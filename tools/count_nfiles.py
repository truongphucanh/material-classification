"""!Warning: Run only one time.
"""

import os
import logging
import tools

TRAIN_FORMAT = 'trainlist0{}.txt'
TEST_FORMAT = 'testlist0{}.txt'

def count_file(folder):
    """Count number of files in a folder.
    
    Arguments:
        folder {string} -- folder directory.
    """
    logger = logging.getLogger()
    if not os.path.exists(folder):
        logger.debug('!Error: Folder {} not found.'.format(folder))
        return 0
    nfiles = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
    if nfiles < 0:
        logger.debug('!Error: nfiles must be positive but it is {}'.format(nfiles))
        nfiles = 0
    return nfiles

def count(train_test_file):
    """Count n_files for each row in train_test_file.
      
    Arguments:
        train_test_file {string} -- train_test_file name (Ex. trainlist01.txt).
    """
    logger = logging.getLogger()
    logger.debug('Running on {}...'.format(train_test_file))
    from_dir = '../train_test/{}'.format(train_test_file)
    lines = []
    total_files = 0
    with open(from_dir) as freader:
        content = freader.readlines()
    content = [x.strip() for x in content]
    folders = [line.split()[0] for line in content]
    for i, folder in enumerate(folders):
        full_dir = '../data/{}'.format(folder)
        if os.path.exists(full_dir):
            nfiles = count_file(full_dir)
            logger.debug('Folder {} has {} files'.format(full_dir, nfiles))
            lines.append('{} {}'.format(content[i], nfiles))
            total_files = total_files + nfiles
        else:
            logger.debug('!Error: folder {} not found'.format(full_dir))
    to_dir = '../temp/{}'.format(train_test_file)
    logger.info('Total for {}: {}'.format(train_test_file, total_files))
    with open(to_dir, 'w') as fwritter:
        for line in lines:
            fwritter.write('{}\n'.format(line))
def main():
    """Main
    """
    logger = tools.get_logger('../logs/count_files.log', file_level = logging.INFO, console_level = logging.DEBUG)
    if not os.path.exists('../temp'):
        os.makedirs('../temp')
    for i in range(0, 6):
        train_file = TRAIN_FORMAT.format(i)
        count(train_file)
        test_file = TEST_FORMAT.format(i)
        count(test_file)

if __name__ == '__main__':
    main()
