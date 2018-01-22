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
    img = image.load_img(img_path,target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    model_extractfeatures = Model(input=model.input, output=model.get_layer('fc2').output)
    fc2_features = model_extractfeatures.predict(x)
    return fc2_features[0]

def get_list_fc2(img_folder, model):
    list_fc2 = []
    print('Runing on folder {}...'.format(img_folder))
    for img_path in glob.glob('{}/*.jpg'.format(img_folder)):
        print('Extracting feature in from {}...'.format(img_path))
        list_fc2.append(get_fc2(img_path, model))
    print('Extracted feature from folder {} success.'.format(img_folder))
    return list_fc2

def get_fc2_and_labels(info_file, model):
    with open(info_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for line in content:
        folder = line.split()[0]
        X_out_file = '../bin/features/keras_vgg16_fc2/{}.pkl'.format(folder)
        if not os.path.exists(X_out_file):
            curr_fc2 = get_list_fc2('../data/{}'.format(folder), model)
            print 'Writing feature to file {}...'.format(X_out_file)
            with open(X_out_file, 'wb') as f:
                pickle.dump(curr_fc2, f)
            print 'Writing success.'        
        else:
            print 'File {} existed.'.format(X_out_file)

def main():
    model = VGG16(weights='imagenet', include_top=True)
    for i in range (1, 6):
        info_train_file = '../train_test/trainlist0{}.txt'.format(str(i))
        info_test_file = '../train_test/testlist0{}.txt'.format(str(i))
        get_fc2_and_labels(info_train_file, model)
        get_fc2_and_labels(info_test_file, model)
    return 0

if __name__ == '__main__':
    main()