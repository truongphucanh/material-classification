# python ./src/

import keras
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
# from keras.models import Model
# from keras import optimizers 
from keras.callbacks import TensorBoard,ModelCheckpoint
# import argparse
from time import time

# Parse arguments
n_classes = 2 # arg
dataset = 'GTOS' # arg
split = 'split_0' # arg
train_dir = './DS_{}/data/{}/train'.format(dataset, split) 
test_dir = './DS_{}/data/{}/test'.format(dataset, split)

# Read data
############ Do some random image transformations to increase the number of training samples
############ Note that we are scaling the image to make all the values between 0 and 1. That's how our pretrained weights have been done too
############ Default batch size is 1 but you can reduce/increase it depending on how powerful your machine is. 
batch_size=1
img_width = 224
img_height = 224
train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

# Build Fine-tune network architecture
## Get Image-Net pre-trained model
vgg16_model = keras.applications.vgg16.VGG16()
print("<VGG16 model>")
vgg16_model.summary()
model = keras.Sequential()
for layer in vgg16_model.layers:
        model.add(layer)
## Remove 1000-classes output layer (last layer)
model.layers.pop()
for layer in model.layers:
        layer.trainable = False
## Add n_classes output layer for our dataset
model.add(keras.layers.Dense(n_classes, activation='softmax'))
print("<Fine-tuned model>")
model.summary()

# Specify model file path
tensorboard = TensorBoard(log_dir="./src/vgg16_experiment/logs/{}.log".format(time()))
filepath = './src/vgg16_experiment/model/fine_tuned_model.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,save_best_only=True,save_weights_only=False, mode='min',period=1)
callbacks_list = [checkpoint, tensorboard]

# Config model and train
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=.0001), metrics=["accuracy"])
num_training_img=100
num_validation_img=20
stepsPerEpoch = num_training_img/batch_size
validationSteps= num_validation_img/batch_size
print("validationSteps: {}".format(validationSteps))
epochs = 2
model.fit_generator(
        train_generator,
        steps_per_epoch=stepsPerEpoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validationSteps,
        callbacks = callbacks_list,
        )
print("[loss, acc] = {}".format(model.evaluate_generator(generator=validation_generator)))