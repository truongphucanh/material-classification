from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import glob
import tools
import logging
import os
import pickle

def get_fc2(img_path, model):
    #logger = tools.get_logger('../logs/feature_extractor.log', logging.INFO, logging.DEBUG)
    img = image.load_img(img_path,target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    model_extractfeatures = Model(input=model.input, output=model.get_layer('fc2').output)
    fc2_features = model_extractfeatures.predict(x)
    return fc2_features[0]

def get_list_fc2(img_folder, model):
    #logger = tools.get_logger('../logs/feature_extractor.log', logging.DEBUG, logging.DEBUG)
    list_fc2 = []
    print('Runing on folder {}...'.format(img_folder))
    for img_path in glob.glob('{}/*.jpg'.format(img_folder)):
        print('Extracting feature in from {}...'.format(img_path))
        list_fc2.append(get_fc2(img_path, model))
    print('Extracted feature from folder {} success.'.format(img_folder))
    return list_fc2

def get_fc2_and_labels(info_file, model):
    #logger = tools.get_logger('../logs/feature_extractor.log', logging.INFO, logging.DEBUG)
    #list_fc2 = []
    #labels = []
    with open(info_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for line in content:
        folder = line.split()[0]
        label = line.split()[1]
        X_out_file = '../bin/vgg16-svm/vgg16/{}_X.pkl'.format(folder)
        y_out_file = '../bin/vgg16-svm/vgg16/{}_y.pkl'.format(folder)
        if not os.path.exists(X_out_file):
            curr_fc2 = get_list_fc2('../data/{}'.format(folder) ,model)
            curr_labels = [label] * np.shape(curr_fc2)[0]
            print 'Writing feature to file {}, {}...'.format(X_out_file, y_out_file)
            with open(X_out_file, 'wb') as f:
                pickle.dump(curr_fc2, f)
            with open(y_out_file, 'wb') as f:
                pickle.dump(curr_labels, f)
            print 'Writing success.'        
        else:
            print 'Feature file {} existed.'.format(X_out_file)
        #list_fc2.extend(curr_fc2)
        #labels.extend([label] * np.shape(curr_fc2)[0])
    #return list_fc2, labels

# def extract_to_file(info_file, model, X_out_file, y_out_file):
#     if not os.path.exists(X_out_file):
#         list_fc2, labels = get_fc2_and_labels(info_file, model)
#         print 'Writing feature to file {}, {}...'.format(X_out_file, y_out_file)
#         with open(X_out_file, 'wb') as f:
#             pickle.dump(list_fc2, f)
#         with open(y_out_file, 'wb') as f:
#             pickle.dump(labels, f)
#         print 'Writing success.'
#     else:
#         print 'Feature file {} existed.'.format(X_out_file)

def main():
    model = VGG16(weights='imagenet', include_top=True)
    
    # info_file = '../train_test/testlist01-copy.txt'
    # get_fc2_and_labels(info_file, model)
    # return 0

    for i in range (1, 6):
        info_train_file = '../train_test/trainlist0{}.txt'.format(str(i))
        info_test_file = '../train_test/testlist0{}.txt'.format(str(i))
        get_fc2_and_labels(info_train_file, model)
        get_fc2_and_labels(info_test_file, model)
        # info_train_file = '../train_test/trainlist0{}.txt'.format(str(i))
        # X_train_file = '../bin//train_test/trainlist0{}_X.pkl'.format(str(i))
        # y_train_file = '../bin//train_test/trainlist0{}_y.pkl'.format(str(i))
        # extract_to_file(info_train_file, model, X_train_file, y_train_file)

        # info_test_file = '../train_test/testlist0{}.txt'.format(str(i))
        # X_test_file = '../bin//train_test/testlist0{}_X.pkl'.format(str(i))
        # y_test_file = '../bin//train_test/testlist0{}_y.pkl'.format(str(i))
        # extract_to_file(info_test_file, model, X_test_file, y_test_file)
    return 0

if __name__ == '__main__':
    main()