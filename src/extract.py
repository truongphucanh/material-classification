# Format: python extract.py \{feature name\} \{dataset\}.
# Ex. python extract.py keras_vgg16_fc2 original

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import sys
import numpy
import kit
import logging
import os
import pickle

VGG16_MODEL = VGG16(weights='imagenet', include_top=True)

def extract_keras_vgg16_fc2(img_path):
    """Extract feature from an image
    
    Arguments:
        img_path {string} -- full path to image file
    
    Returns:
        numpy.array -- vector of feature
    """
    img = image.load_img(img_path,target_size=(224, 224))
    x = image.img_to_array(img)
    x = numpy.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = VGG16_MODEL.predict(x)
    model_extractfeatures = Model(input=VGG16_MODEL.input, output=VGG16_MODEL.get_layer('fc2').output)
    fc2_features = model_extractfeatures.predict(x)
    return numpy.array(fc2_features[0])

def extract_all(feature, dataset):
    """Extract feature for all images in dataset.
    
    Arguments:
        feature {string} -- feature name (Ex. "keras_vgg16_fc2").
        dataset {string} -- name of dataset (Ex. "original")
    """
    logger = logging.getLogger()
    data_folder = '../data/{}'.format(dataset)
    for root, dirs, files in os.walk(data_folder):
        for filename in [f for f in files if f.endswith(".jpg")]:
            jpg = os.path.join(root, filename)
            pkl = jpg.replace('data', 'bin/features/{}-{}'.format(feature, dataset)).replace('.jpg', '.pkl')
            pkl = pkl.replace('/{}'.format(dataset),'')
            feature_folder = root.replace('data', 'bin/features/keras_vgg16_fc2')
            if os.path.exists(pkl):
                logger.debug('Feature file {} existed.'.format(pkl))
            else:
                if not os.path.exists(feature_folder):
                    os.makedirs(feature_folder)
                logger.debug(jpg)
                if feature == "keras_vgg16_fc2":
                    X = extract_keras_vgg16_fc2(jpg)
                else:
                    logger.warn("!Warning: Function to extract {} is not defined".format(feature))
                    return
                with open(pkl, 'wb') as f:
                    pickle.dump(X, f)

def main(argv):
    kit.config()
    if len(argv) < 3:
        print("Missing arguments. Please try again.")
        print("Format: python extract.py \{feature name\} \{dataset\}.")
        print("Ex. python extract.py keras_vgg16_fc2 edges.")
        return
    feature = argv[1]
    dataset = argv[2]
    logger = kit.get_logger(file_name="../logs/extract.log")
    logger.info("*" * 50)
    logger.info("feature: {}".format(feature))
    logger.info("dataset: {}".format(dataset))
    extract_all(feature, dataset)

if __name__ == '__main__':
    main(sys.argv)
