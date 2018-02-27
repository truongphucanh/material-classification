import pickle
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

def get_fc2(img_path, model):
    img = image.load_img(img_path,target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    model_extractfeatures = Model(input=model.input, output=model.get_layer('fc2').output)
    fc2_features = model_extractfeatures.predict(x)
    return fc2_features[0]

model = VGG16(weights='imagenet', include_top=True)
feat = get_fc2('2.jpg', model)
with open('feat2.pkl', 'wb') as f:
    pickle.dump(feat, f)
with open('feat2.pkl', 'rb') as f:
    feat = pickle.load(f)
print(feat)