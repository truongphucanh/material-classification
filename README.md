# Material Classification

## Setup

1. Run **./setup/install_python.cmd** to install python 3.

2. Run command **python** in command line to check whether install sucess or not.

3. Run **./setup/install_packages.cmd** to install python's packages.

4. Run **python ./setup/test/tensorflow_test.py** to validate your tensorflow installation. If "Hello, TensorFlow!" is printed, your installation is success.

5. Run **python ./setup/test/keras_test.py** to validate your keras installation. If "Hello Keras!" is printed, your installation is success.

## Dataset

You can find it here.

## Requirements

Python 3.6

## To get texture images for a dataset

1. Make sure you have original dataset in folder **./DS_<dataset_name>/data/original**.

2. Run the following command.

```batch
python ./tools/create_data_folders.py <dataset_name> texture
python ./tools/get_data_filenames.py <dataset_name> original
```

1. Open Matlab.exe, change working folder to this project folder.

2. In get_texture.m edit txt file name into <dataset_name>_original.txt.

3. Run get_texture.m.

4. Open folder ./DS_<dataset_name>/data/texture to check
