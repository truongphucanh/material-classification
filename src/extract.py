# Plesase change current working directory to root folder (of this project) before run source
# Format: python extract.py <feature name> <dataset_name> <dataset_type>
# Ex. python ./src/extract.py keras_vgg16_fc2 FMD edges

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import sys
import numpy
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

def extract_all(feature_name, dataset_name, dataset_type):
    """Extract feature for all images in dataset.
    
    Arguments:
        feature_name {string} -- feature name (Ex. "keras_vgg16_fc2").
        dataset {string} -- name of dataset (Ex. "original")
    """
    data_folder = './DS_{}/data/{}'.format(dataset_name, dataset_type)
    print(data_folder)
    for root, dirs, files in os.walk(data_folder):
        for filename in [f for f in files if f.endswith(".jpg")]:
            jpg = os.path.join(root, filename).replace("\\", "/")
            pkl = jpg.replace('data', 'result/features').replace('.jpg', '.pkl')
            feature_folder = root.replace('data', 'result/features')
            if os.path.exists(pkl):
                print('Feature file {} existed.'.format(pkl))
            else:
                if not os.path.exists(feature_folder):
                    os.makedirs(feature_folder)
                print(jpg)
                if feature_name == "keras_vgg16_fc2":
                    X = extract_keras_vgg16_fc2(jpg)
                else:
                    logger.warn("!Warning: Function to extract {} is not defined".format(feature_name))
                    return
                with open(pkl, 'wb') as f:
                    pickle.dump(X, f)

def main(argv):
    if len(argv) < 4:
        print("Missing arguments. Please try again.")
        print("Format: python extract.py <feature_name> <dataset_name> <dataset_type>")
        print("Ex. python extract.py keras_vgg16_fc2 FMD edges")
        return
    feature_name = argv[1]
    dataset_name = argv[2]
    dataset_type = argv[3]
    extract_all(feature_name, dataset_name, dataset_type)

if __name__ == '__main__':
    main(sys.argv)
