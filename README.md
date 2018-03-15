# Dataset

You can find it here.

# Requirements

Python 3.6

## To get texture images for a dataset

1. Make sure you have original dataset in folder **./DS_<dataset_name>/data/original**
2. Run the following command
```batch
python ./tools/create_data_folders.py <dataset_name> texture
python ./tools/get_data_filenames.py <dataset_name> original
```
3. Open Matlab.exe, change working folder to this project folder
4. In get_texture.m edit txt file name into <dataset_name>_original.txt
5. Run get_texture.m
6. Open folder ./DS_<dataset_name>/data/texture to check
