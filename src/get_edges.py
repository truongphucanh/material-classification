""" Get edges from original dataset """
import os
import logging
import cv2
import tools

ORIGINAL_DATA_FOLDER = '../data/original'
EDGES_DATA_FOLDER = '../data/edges'
LOG_FILE = '../logs/get_edges.log'

def main():
    """Entry point.
    """
    tools.config()
    logger = tools.get_logger(LOG_FILE, logging.INFO, logging.DEBUG)
    for dirpath, _, filenames in os.walk(ORIGINAL_DATA_FOLDER):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            original_image_path = os.path.join(dirpath, filename)
            edge_image_path = original_image_path.replace('original', 'edges')
            edge_folder = dirpath.replace('original', 'edges')
            if not os.path.exists(edge_folder):
                logger.debug('Creating folder {} ...'.format(edge_folder))
                os.makedirs(edge_folder)
            original_image = cv2.imread(original_image_path, 0)
            edge_image = cv2.Canny(original_image, 100, 200)
            logger.debug('Writing image {}'.format(edge_image_path))
            cv2.imwrite(edge_image_path, edge_image)
            
    print 'Done'

if __name__ == '__main__':
    main()
