# Material Classification

## Authors

1. Truong Phuc Anh - 14520040@gm.uit.edu.vn

## Requirements

1. Python 3.6

2. Matlab

## Setup

1. Run **./setup/install_python.cmd** to install python 3.

2. Run command **python** in command line to check whether install sucess or not.

3. Run **./setup/install_packages.cmd** to install python's packages.

4. Run **python ./setup/test/tensorflow_test.py** to validate your tensorflow installation. If "Hello, TensorFlow!" is printed, your installation is success.

5. Run **python ./setup/test/keras_test.py** to validate your keras installation. If "Hello Keras!" is printed, your installation is success.

## Dataset

Download 2 datasets using link below, extra it in **./src/DS_<dataset name>/original**.

Re-format dataset into **./src/DS_<dataset name>/original/<classes name>**

1. GTOS dataset: http://eceweb1.rutgers.edu/vision/gts/download.html

2. FMD dataset: https://people.csail.mit.edu/celiu/CVPR2010/FMD/

## Get texture images for a dataset

1. Make sure you have original dataset in folder **./DS_<dataset_name>/data/original**.

2. Run the following command.

```batch
python ./src/tools/create_data_folders.py <dataset_name> texture
python ./src/get_data_filenames.py <dataset_name> original
```

1. Open Matlab.exe, change working folder to **./src**.

2. In get_texture.m edit txt file name into <dataset_name>_original.txt.

3. Run get_texture.m.

4. Open folder ./DS_<dataset_name>/data/texture to check

## Get edges images

1. Make sure you have original dataset in folder **./DS_<dataset_name>/data/original**.

2. Run the following command.

```batch
python ./src/local-feature-extractor/extract_candy_edge.py <dataset_name> 
```

## Train and test with SVM

1. Make sure you have original, edges and texture dataset in folder **./DS_<dataset_name>/data**.

2. Configs your own SVM kernel parameter in file **./config/models_config.csv**

3. Run the following command.

```batch
./src/run-FMD.cmd
./src/run-GTOS.cmd
```

## Fine-tuning with VGG16

1. Only re-train the last layer. Run the following commands:

```batch
python .\src\vgg16_experiment\finetune.py GTOS 1 39
python .\src\vgg16_experiment\finetune.py GTOS 2 39
python .\src\vgg16_experiment\finetune.py GTOS 3 39
python .\src\vgg16_experiment\finetune.py GTOS 4 39
python .\src\vgg16_experiment\finetune.py GTOS 5 39

python .\src\vgg16_experiment\finetune.py FMD 1 10
python .\src\vgg16_experiment\finetune.py FMD 2 10
python .\src\vgg16_experiment\finetune.py FMD 3 10
python .\src\vgg16_experiment\finetune.py FMD 4 10
python .\src\vgg16_experiment\finetune.py FMD 5 10
```

2. Re-train all layers. Run the following commands

```batch
python .\src\vgg16_experiment\finetune_full.py GTOS 1 39
python .\src\vgg16_experiment\finetune_full.py GTOS 2 39
python .\src\vgg16_experiment\finetune_full.py GTOS 3 39
python .\src\vgg16_experiment\finetune_full.py GTOS 4 39
python .\src\vgg16_experiment\finetune_full.py GTOS 5 39

python .\src\vgg16_experiment\finetune_full.py FMD 1 10
python .\src\vgg16_experiment\finetune_full.py FMD 2 10
python .\src\vgg16_experiment\finetune_full.py FMD 3 10
python .\src\vgg16_experiment\finetune_full.py FMD 4 10
python .\src\vgg16_experiment\finetune_full.py FMD 5 10
```

## How to view result?

Results are stored in **./<dataset_name>/result/test**

Including: accuracy, miss sample, classifiers, confusion matrix, etc.