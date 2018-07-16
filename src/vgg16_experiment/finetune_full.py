# python .\src\vgg16_experiment\finetune.py GTOS 0 39

import keras
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
# from keras.models import Model
# from keras import optimizers 
from keras.callbacks import TensorBoard,ModelCheckpoint
# import argparse
from time import time
import os
import sys

# Parse arguments
dataset = sys.argv[1] # arg
split = sys.argv[2] # arg
n_classes = int(sys.argv[3]) # arg

train_dir = './DS_{}/data/split_{}/train'.format(dataset, split) 
test_dir = './DS_{}/data/split_{}/test'.format(dataset, split)

# Read data
############ Do some random image transformations to increase the number of training samples
############ Note that we are scaling the image to make all the values between 0 and 1. That's how our pretrained weights have been done too
############ Default batch size is 1 but you can reduce/increase it depending on how powerful your machine is. 
batch_size=1
img_width = 224
img_height = 224
train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = image.ImageDataGenerator(rescale=1. / 255)
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
        layer.trainable = True
## Add n_classes output layer for our dataset
model.add(keras.layers.Dense(n_classes, activation='softmax'))
print("<Fine-tuned model>")
model.summary()

# Specify model file path
tensorboard = TensorBoard(log_dir="./src/vgg16_experiment/logs/{}.log".format(time()))
model_folder = './src/vgg16_experiment/models'
filepath = '{}/fine_tuned_model_{}.h5'.format(model_folder, split)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,save_best_only=True,save_weights_only=False, mode='min',period=1)
callbacks_list = [checkpoint, tensorboard]

# Config model and train
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=.0001), metrics=["accuracy"])
num_training_img = 24954
num_validation_img = 9151
#stepsPerEpoch = num_training_img/batch_size
stepsPerEpoch = 1000
validationSteps= num_validation_img/batch_size
print("validationSteps: {}".format(validationSteps))
epochs = 10
model.fit_generator(
        train_generator,
        steps_per_epoch=stepsPerEpoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validationSteps,
        callbacks = callbacks_list,
        )
result_folder = "./src/vgg16_experiment/result/{}".format(dataset)
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
accuracy_file = "{}/acc.csv".format(result_folder)
evaluater = model.evaluate_generator(generator=validation_generator)
loss = evaluater[0]
acc = evaluater[1]
with open(accuracy_file, "a") as fw:
        fw.write("{}, {}, {}\n".format(split, loss, acc))
print("[loss, acc] = {}".format(model.evaluate_generator(generator=validation_generator)))