# python ./tools/create_data_folders.py <dataset_name> <dataset_type>
# python ./tools/create_data_folders.py FMD texture

import os
import sys

dataset_name = sys.argv[1]
dataset_type = sys.argv[2]

data_folder = './DS_{}/data/original'.format(dataset_name)
for root, dirs, files in os.walk(data_folder):
    for filename in [f for f in files if f.endswith(".jpg")]:
        jpg = root
        texture = jpg.replace('original', dataset_type)
        if os.path.exists(texture):
            continue
        print(texture)
        os.makedirs(texture)