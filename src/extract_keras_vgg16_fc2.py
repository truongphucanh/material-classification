""" Extracting vgg16 fc2 using keras framework and tensorflow as backend"""
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import sys
import numpy as np
import glob
import tools
import logging
import os
import pickle

def extract_from(img_path, model):
    """Extract feature from an image
    
    Arguments:
        img_path {string} -- full path to image file
        model {Model} -- VGG16 model
    
    Returns:
        list -- vector of feature
    """

    img = image.load_img(img_path,target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    model_extractfeatures = Model(input=model.input, output=model.get_layer('fc2').output)
    fc2_features = model_extractfeatures.predict(x)
    return fc2_features[0]

def extract_all(dataset_name, model):
    """Extract feature for all images in dataset.
    
    Arguments:
        dataset_name {string} -- name of dataset (Ex. 'original')
        model {Model} -- vgg16 model used to extract feature
    """
    logger = logging.getLogger()
    data_folder = '../data/{}'.format(dataset_name)
    for dirpath, _, filenames in os.walk(data_folder):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            image_path = os.path.join(dirpath, filename)
            feature_path = image_path.replace('data', 'bin/features/keras_vgg16_fc2').replace('.jpg', '.pkl')
            feature_folder = dirpath.replace('data', 'bin/features/keras_vgg16_fc2')
            if os.path.exists(feature_path):
                logger.debug('Feature file {} existed.'.format(feature_path))
            else:
                if not os.path.exists(feature_folder):
                    os.makedirs(feature_folder)
                    logger.info('Running on folder'.format(feature_folder))
                logger.debug('Extracting from {}...'.format(image_path))
                feature = extract_from(image_path, model)
                with open(feature_path, 'wb') as f:
                    pickle.dump(feature, f)

def main(argv):
    tools.config()
    if len(argv) < 2:
        print('Missing arguments: extract_keras_vgg16_fc2.py `dataset_name`. Ex. `python train.py original`')
        return
    model = VGG16(weights='imagenet', include_top=True)
    dataset_name = argv[1]
    log_file = '../logs/extract-keras-vgg16-fc2-{}.log'.format(dataset_name)
    logger = tools.get_logger(log_file, logging.INFO, logging.DEBUG)
    extract_all(dataset_name, model)
    return 0

if __name__ == '__main__':
    main(sys.argv)